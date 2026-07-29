from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, TypedDict

import numpy as np
import torch

import triton_python_backend_utils as pb_utils
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

class Span(TypedDict):
    tag: str
    begin: int
    end: int
    value: str

class Prediction(TypedDict):
    token_id: int
    value: str
    label: str
    begin: int
    end: int

class SpanResult(TypedDict):
    spans: list[Span]
    predictions: list[Prediction]

def _decode_spans(text: str, offsets: list[tuple[int, int]], pred_ids: list[int], id2label: dict) -> list[Span]:
    """Merge token level BIO predictions into spans using the offset mapping"""
    spans: list[Span] = []
    cur: Span | None = None

    def close() -> None:
        nonlocal cur
        if cur is not None:
            cur["value"] = text[cur["begin"]:cur["end"]]
            spans.append(cur)
        cur = None

    for (start, end), pred_id in zip(offsets, pred_ids):
        # special/pad tokens carry an empty (0, 0) offset
        if start == end:
            close()
            continue

        tag = id2label.get(pred_id, id2label.get(str(pred_id), "O"))
        if tag == "O":
            close()
            continue

        prefix, _, label = tag.partition("-")
        if prefix == "B" or cur is None or cur["tag"] != label:
            close()
            cur = Span(tag=label, begin=start, end=end, value="")
        else:
            cur["end"] = end

    close()
    return spans

def _decode_predictions(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: list[int],
    offsets: list[tuple[int, int]],
    pred_ids: list[int],
    id2label: dict,
) -> list[Prediction]:
    """Per token subword, its character offsets and its predicted label"""
    special_ids = set(tokenizer.all_special_ids)
    predictions: list[Prediction] = []
    for token_id, (start, end), pred_id in zip(token_ids, offsets, pred_ids):
        if token_id in special_ids:
            continue

        label = id2label.get(pred_id, id2label.get(str(pred_id), "O"))
        predictions.append(
            Prediction(
                token_id=token_id,
                value=tokenizer.convert_ids_to_tokens(token_id),
                label=label,
                begin=start,
                end=end,
            )
        )
    return predictions

class TritonPythonModel:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel

    def initialize(self, args: dict[str, str]) -> None:
        self.logger = pb_utils.Logger

        # Read from config.pbtxt
        weight_dir = str(Path(__file__).resolve().parent / "model")
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        def get_param(name: str, default_value: Any) -> str:
            """Read a `parameters` entry from config.pbtxt"""
            param = params.get(name, {"string_value": str(default_value)})
            return param["string_value"]

        self.logger.log_info(f"Loading {weight_dir} with params {params}")

        # Tokenizer
        ## Init tokenizer params
        self.tokenizer_max_length = int(get_param("TOKENIZER_MAX_LENGTH", "512"))
        tokenizer_padding_side = get_param("TOKENIZER_PADDING_SIDE", "right")
        tokenizer_additional_params = get_param("TOKENIZER_ADDITIONAL_PARAMS", "{}")
        tokenizer_additional_params = json.loads(tokenizer_additional_params)

        ## Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            weight_dir,
            use_fast=True,
            trust_remote_code=True,
            padding_side=tokenizer_padding_side,
            **tokenizer_additional_params
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        model_kwargs: dict[str, Any] = dict()
        ## Init model params
        model_attn_implementation = get_param("MODEL_ATTN_IMPLEMENTATION", "none")
        if model_attn_implementation.lower() != "none":
            model_kwargs["attn_implementation"] = model_attn_implementation

        if get_param("MODEL_USE_FP16", "false").lower() == "true":
            model_kwargs["dtype"] = torch.float16
        elif get_param("MODEL_USE_BF16", "false").lower() == "true":
            model_kwargs["dtype"] = torch.bfloat16

        model_additional_params = get_param("MODEL_ADDITIONAL_PARAMS", "{}")
        model_additional_params = json.loads(model_additional_params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Load Model
        self.model = AutoModelForTokenClassification.from_pretrained(
            weight_dir,
            trust_remote_code=True,
            **model_kwargs,
            **model_additional_params
        ).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

        ## Optional torch.compile, skipped when the backend cannot compile the model
        if get_param("MODEL_USE_TORCH_COMPILE", "false").lower() == "true":
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                self.logger.log_info("PyTorch 2.0 compilation enabled")
            except Exception as e:
                self.logger.log_warn(f"Could not apply torch.compile: {e}")

        ## Additional Params
        self.inference_batch_size = int(get_param("INFERENCE_BATCH_SIZE", "16"))

    def _parse_request(self, request: pb_utils.InferenceRequest) -> list[str]:
        """Read texts (list[str]) from request"""
        input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
        if input_tensor is None:
            raise ValueError("Input tensor is None")
        texts = [t.decode("utf-8") for t in input_tensor.as_numpy().flatten()]
        return texts

    def _parse_requests(self, requests: list[pb_utils.InferenceRequest]) -> tuple[list[str], list[int]]:
        """
        Args:
            requests: received batched requests
        Returns:
            list[str]: flattened input text list
            list[int]: request index for each item
        """
        texts = []
        indices = []
        for i, request in enumerate(requests):
            request_texts = self._parse_request(request)
            texts.extend(request_texts)
            indices.extend([i]*len(request_texts))
        return texts, indices

    @torch.inference_mode()
    def _infer(self, texts: list[str]) -> list[SpanResult]:
        """Predict spans and per token predictions"""
        inputs = self.tokenizer(
            texts,
            max_length=self.tokenizer_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # offset_mapping is not a model input, keep it for decoding only
        offsets_mapping = inputs.pop("offset_mapping").tolist()
        input_ids = inputs["input_ids"].tolist()
        inputs.to(self.device)

        outputs = self.model(**inputs)
        pred_ids = outputs.logits.argmax(dim=-1).tolist()

        results: list[SpanResult] = []
        for text, offsets, token_ids, item_pred_ids in zip(texts, offsets_mapping, input_ids, pred_ids):
            results.append(
                {
                    "spans": _decode_spans(text, offsets, item_pred_ids, self.id2label),
                    "predictions": _decode_predictions(
                        self.tokenizer, token_ids, offsets, item_pred_ids, self.id2label
                    ),
                }
            )
        return results

    def _predict(self, texts: list[str]) -> list[SpanResult]:
        """Predict spans for texts, chunked by inference_batch_size"""
        results: list[SpanResult] = []
        for i in range(0, len(texts), self.inference_batch_size):
            batch_texts = texts[i : i + self.inference_batch_size]
            results.extend(self._infer(batch_texts))
        return results

    def execute(self, requests: list[pb_utils.InferenceRequest]) -> list[pb_utils.InferenceResponse]:
        # Parse Requests
        texts, indices = self._parse_requests(requests)
        self.logger.log_info(f"Received {len(requests)} requests with {len(texts)} texts")

        # Inference
        start = time.time()
        results = self._predict(texts)
        end = time.time()

        duration_ms = (end - start) * 1000
        self.logger.log_info(f"Inference complete in {duration_ms:.2f}ms")

        spans = np.array(
            [json.dumps(r["spans"], ensure_ascii=False).encode("utf-8") for r in results],
            dtype=object,
        )
        predictions = np.array(
            [json.dumps(r["predictions"], ensure_ascii=False).encode("utf-8") for r in results],
            dtype=object,
        )
        request_indices = np.array(indices, dtype=np.int64)

        # Split results back per request
        responses: list[pb_utils.InferenceResponse] = []
        for i in range(len(requests)):
            mask = request_indices == i
            spans_tensor = pb_utils.Tensor("spans", spans[mask])
            predictions_tensor = pb_utils.Tensor("predictions", predictions[mask])
            responses.append(pb_utils.InferenceResponse(output_tensors=[spans_tensor, predictions_tensor]))

        return responses

    def finalize(self) -> None:
        self.logger.log("Finalizing", self.logger.INFO)
        del self.model
        del self.tokenizer
