from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, TypedDict

import numpy as np

import triton_python_backend_utils as pb_utils
import vllm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM

class Prediction(TypedDict):
    label: int
    scores: list[float]

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
        tokenizer_padding_side = get_param("TOKENIZER_PADDING_SIDE", "right")

        ## Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            weight_dir,
            padding_side=tokenizer_padding_side
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        ## Init engine params
        self.max_model_len = int(get_param("VLLM_MAX_MODEL_LEN", "512"))

        ## Inputs are truncated below max_model_len to keep the engine from hanging
        max_model_len_truncation_offset = int(get_param("VLLM_MAX_MODEL_LEN_TRUNCATION_OFFSET", "16"))
        if max_model_len_truncation_offset > self.max_model_len:
            raise ValueError(
                f"VLLM_MAX_MODEL_LEN_OFFSET must be smaller than VLLM_MAX_MODEL_LEN "
                f"({max_model_len_truncation_offset}>{self.max_model_len})"
            )
        self.tokenizer_max_length = self.max_model_len - max_model_len_truncation_offset

        engine_additional_params = json.loads(get_param("VLLM_ADDITIONAL_OPTIONS", "{}"))
        engine_kwargs: dict[str, Any] = {
            "max_model_len": self.max_model_len,
            "dtype": get_param("VLLM_DTYPE", "auto"),
            "gpu_memory_utilization": float(get_param("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            "max_num_seqs": int(get_param("VLLM_MAX_NUM_SEQS", "8")),
            "runner": "pooling",
            **engine_additional_params
        }

        ## Load Model
        self.model = LLM(model=weight_dir, tokenizer=weight_dir, **engine_kwargs)

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

    def _truncate(self, texts: list[str]) -> list[str]:
        """Encode -> truncate -> decode so no input exceeds the engine's max_model_len"""
        encoded = self.tokenizer(
            texts,
            max_length=self.tokenizer_max_length,
            truncation=True,
        )
        return self.tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=True)

    def _infer(self, texts: list[str]) -> np.ndarray:
        """Predict scores"""
        outputs = self.model.classify(texts)
        scores: np.ndarray = np.array([output.outputs.probs for output in outputs], dtype=np.float32)
        return scores

    def _predict(self, texts: list[str]) -> list[Prediction]:
        """Predict labels and scores for texts, batched by the engine itself"""
        scores = self._infer(self._truncate(texts))
        pred_labels: np.ndarray = np.argmax(scores, axis=1)

        predictions: list[Prediction] = [
            {"label": int(label), "scores": item_scores.tolist()}
            for label, item_scores in zip(pred_labels, scores)
        ]
        return predictions

    def execute(self, requests: list[pb_utils.InferenceRequest]) -> list[pb_utils.InferenceResponse]:
        # Parse Requests
        texts, indices = self._parse_requests(requests)
        self.logger.log_info(f"Received {len(requests)} requests with {len(texts)} texts")

        # Inference
        start = time.time()
        predictions = self._predict(texts)
        end = time.time()

        duration_ms = (end - start) * 1000
        self.logger.log_info(f"Inference complete in {duration_ms:.2f}ms")

        labels = np.array([p["label"] for p in predictions], dtype=np.int32)
        scores = np.array([p["scores"] for p in predictions], dtype=np.float32)
        request_indices = np.array(indices, dtype=np.int64)

        # Split results back per request
        responses: list[pb_utils.InferenceResponse] = []
        for i in range(len(requests)):
            mask = request_indices == i
            label_tensor = pb_utils.Tensor("label", labels[mask])
            scores_tensor = pb_utils.Tensor("scores", scores[mask])
            responses.append(pb_utils.InferenceResponse(output_tensors=[label_tensor, scores_tensor]))

        return responses

    def finalize(self) -> None:
        self.logger.log("Finalizing", self.logger.INFO)
        del self.model
        del self.tokenizer
