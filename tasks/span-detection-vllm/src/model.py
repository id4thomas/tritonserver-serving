from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, TypedDict

import numpy as np

import triton_python_backend_utils as pb_utils
import vllm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, TokensPrompt

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
    model: LLM

    def initialize(self, args: dict[str, str]) -> None:
        self.logger = pb_utils.Logger
        self.logger.log_info(f"vLLM version: {vllm.__version__}")

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
        self.tokenizer_max_length = int(get_param("TOKENIZER_MAX_LENGTH", "256"))
        self.vllm_max_model_len = self.tokenizer_max_length
        tokenizer_padding_side = get_param("TOKENIZER_PADDING_SIDE", "right")
        tokenizer_additional_params = get_param("TOKENIZER_ADDITIONAL_PARAMS", "{}")
        tokenizer_additional_params = json.loads(tokenizer_additional_params)

        # Apply max model len truncation offset
        max_length_truncation_offset = int(get_param("TOKENIZER_MAX_LENGTH_TRUNCATION_OFFSET", "16"))
        if max_length_truncation_offset > self.tokenizer_max_length:
            raise ValueError(
                f"TOKENIZER_MAX_LENGTH_TRUNCATION_OFFSET must be smaller than TOKENIZER_MAX_LENGTH "
                f"({max_length_truncation_offset}>{self.tokenizer_max_length})"
            )
        self.tokenizer_max_length -= max_length_truncation_offset

        ## Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            weight_dir,
            padding_side=tokenizer_padding_side,
            **tokenizer_additional_params
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # vLLM engine
        ## Init vllm params
        vllm_gpu_util =  float(get_param("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        vllm_max_num_seqs = int(get_param("VLLM_MAX_NUM_SEQS", "8"))
        vllm_dtype = get_param("VLLM_DTYPE", "auto")

        vllm_additional_params = json.loads(get_param("VLLM_ADDITIONAL_OPTIONS", "{}"))
        vllm_kwargs: dict[str, Any] = {
            "max_model_len": self.vllm_max_model_len,
            "dtype": vllm_dtype,
            "gpu_memory_utilization": vllm_gpu_util,
            "max_num_seqs": vllm_max_num_seqs,
            "runner": "pooling",
            "trust_remote_code": True,
            **vllm_additional_params
        }

        ## Load Engine
        self.model = LLM(model=weight_dir, tokenizer=weight_dir, **vllm_kwargs)

        self.id2label = self.model.llm_engine.vllm_config.model_config.hf_config.id2label

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

    def _tokenize(self, texts: list[str]) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
        """Encode with truncation so no input exceeds the engine's max_model_len"""
        encoded = self.tokenizer(
            texts,
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_offsets_mapping=True,
        )
        return encoded["input_ids"], encoded["offset_mapping"]

    def _infer(self, token_ids_list: list[list[int]]) -> list[list[int]]:
        """Predict a label id per token"""
        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in token_ids_list]
        outputs = self.model.encode(prompts, pooling_task="token_classify", use_tqdm=False)
        return [output.outputs.data.argmax(dim=-1).tolist() for output in outputs]

    def _predict(self, texts: list[str]) -> list[SpanResult]:
        """Predict spans and per token predictions, batched by the engine itself"""
        token_ids_list, offsets_list = self._tokenize(texts)
        pred_ids_list = self._infer(token_ids_list)

        results: list[SpanResult] = []
        for text, token_ids, offsets, pred_ids in zip(texts, token_ids_list, offsets_list, pred_ids_list):
            results.append(
                {
                    "spans": _decode_spans(text, offsets, pred_ids, self.id2label),
                    "predictions": _decode_predictions(
                        self.tokenizer, token_ids, offsets, pred_ids, self.id2label
                    ),
                }
            )
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
