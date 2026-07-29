from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, TypedDict

import numpy as np
import torch

import triton_python_backend_utils as pb_utils
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

class Prediction(TypedDict):
    label: int
    scores: list[float]

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
        self.tokenizer_max_length = int(get_param("TOKENIZER_MAX_LENGTH", "256"))
        tokenizer_padding_side = get_param("TOKENIZER_PADDING_SIDE", "right")
        tokenizer_additional_params = get_param("TOKENIZER_ADDITIONAL_PARAMS", "{}")
        tokenizer_additional_params = json.loads(tokenizer_additional_params)
        
        ## Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            weight_dir,
            padding_side=tokenizer_padding_side,
            **tokenizer_additional_params
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model
        model_kwargs: dict[str, Any] = dict()
        ## Init model params
        model_attn_implementation = get_param("MODEL_ATTN_IMPLEMENTATION", "none")
        if model_attn_implementation!="none":
            model_kwargs["attn_implementation"]=model_attn_implementation

        model_additional_params = get_param("MODEL_ADDITIONAL_PARAMS", "{}")
        model_additional_params = json.loads(model_additional_params)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## Load Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            weight_dir,
            **model_kwargs,
            **model_additional_params
        ).to(self.device)
        self.model.eval()

        ## Models without a pad token of their own (e.g. decoder-based classifiers) reject
        ## batched input unless the pad id is set on the config as well as the tokenizer
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

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
    def _infer(self, texts: list[str]) -> np.ndarray:
        """Predict scores"""
        inputs = self.tokenizer(
            texts,
            max_length=self.tokenizer_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs.to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        scores: np.ndarray = torch.nn.functional.softmax(logits.to(torch.float32), dim=-1).cpu().numpy()
        return scores

    def _predict(self, texts: list[str]) -> list[Prediction]:
        """Predict labels and scores for texts, chunked by inference_batch_size"""
        all_scores: list[np.ndarray] = []
        for i in range(0, len(texts), self.inference_batch_size):
            batch_texts = texts[i : i + self.inference_batch_size]
            all_scores.append(self._infer(batch_texts))
        scores = np.concatenate(all_scores, axis=0)
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