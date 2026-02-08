from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import vllm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM


class TritonPythonModel:
    tokenizer: PreTrainedTokenizerBase
    model: LLM

    def initialize(self, args: dict[str, str]) -> None:
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional.
        This function allows the model to initialize any state associated with this model.

        Args:
            args (dict[str, str]) : Both keys and values are strings.
            The dictionary keys and values are:
            * model_config: A JSON string containing the model configuration
            * model_instance_kind: A string containing model instance kind
            * model_instance_device_id: A string containing model instance device ID
            * model_repository: Model repository path
            * model_version: Model version
            * model_name: Model name
        """
        self.logger = pb_utils.Logger
        self.logger.log_info(f"vLLM version: {vllm.__version__}")

        # model_dir = str(Path(__file__).resolve().parent / "model")
        model_dir = "/tmp/model"
        self.logger.log_info(f"Loading model from {model_dir}")

        model_config = json.loads(args["model_config"])
        self.logger.log_info(f"Model config: {model_config}")
        if "max_batch_size" in model_config:
            self.trt_max_batch_size = model_config["max_batch_size"]
        else:
            self.logger.log_warn("TRITON MAX_BATCH_SIZE not found in model_config, using default value 1")
            self.trt_max_batch_size = 1
        self.logger.log_info(f"TRITON MAX_BATCH_SIZE set to {self.trt_max_batch_size}")

        params = model_config.get("parameters", {})

        def get_param(name: str, default_value: Any) -> str:
            param = params.get(name, {"string_value": str(default_value)})
            return param["string_value"]

        # config.pbtxt params
        self.max_model_len = int(get_param("VLLM_MAX_MODEL_LEN", 512))
        ## apply offset to max_model_len to prevent engine hanging
        self.max_model_len_offset = int(get_param("VLLM_MAX_MODEL_LEN_OFFSET", 16))
        if self.max_model_len_offset > self.max_model_len:
            raise ValueError(
                f"VLLM_MAX_MODEL_LEN_OFFSET must be smaller than VLLM_MAX_MODEL_LEN {self.max_model_len_offset}>{self.max_model_len}"
            )
        self.tokenizer_max_model_len = self.max_model_len - self.max_model_len_offset
        self.max_num_seqs = int(get_param("VLLM_MAX_NUM_SEQS", 8))
        self.dtype = get_param("VLLM_DTYPE", "auto")
        self.gpu_memory_utilization = float(get_param("VLLM_GPU_MEMORY_UTILIZATION", 0.9))
        self.vllm_additional_options = json.loads(get_param("VLLM_ADDITIONAL_OPTIONS", "{}"))

        self.tokenizer_padding_side = get_param("TOKENIZER_PADDING_SIDE", "right")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log_info(f"Using device: {self.device}")

        engine_kwargs: dict[str, Any] = {
            "max_model_len": self.max_model_len,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "runner": "pooling",
            **self.vllm_additional_options,
        }

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side=self.tokenizer_padding_side)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LLM(model=model_dir, tokenizer=model_dir, **engine_kwargs)

    def _parse_requests(self, requests: list[pb_utils.InferenceRequest]) -> list[str]:
        """Triton 요청에서 텍스트 데이터를 추출합니다.
        max_length 초과 오류 방지를 위해 encode -> truncate -> decode과정을 거칩니다
        """
        all_texts: list[str] = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            if input_tensor is None:
                raise ValueError("Input tensor is None")
            texts = [t.decode("utf-8") for t in input_tensor.as_numpy().flatten()]
            all_texts.extend(texts)

        # Encode -> Truncate -> Decode
        tokenized_output = self.tokenizer(
            all_texts,
            max_length=self.tokenizer_max_model_len,
            truncation=True,
            padding="max_length",  # Pad to max_length
            return_tensors="pt",  # Return PyTorch tensors
        )
        all_texts = self.tokenizer.batch_decode(tokenized_output["input_ids"], skip_special_tokens=True)

        self.logger.log_warn(f"Parsing {len(requests)} requests total {len(all_texts)} texts")
        return all_texts

    def _format_outputs(self, outputs):
        """Converts VLLM ClassificationRequestOutput to numpy arrays."""
        all_predictions = []
        all_probabilities = []  # (batch, num_labels)

        for output in outputs:
            probs = output.outputs.probs
            predictions = np.argmax(probs, axis=-1).astype(np.int32)

            all_probabilities.append(probs)
            all_predictions.append(predictions)

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        return all_predictions, all_probabilities

    def _create_responses(
        self, requests: list[pb_utils.InferenceRequest], all_predictions: np.ndarray, all_probabilities: np.ndarray
    ) -> list[pb_utils.InferenceResponse]:
        """추론 결과를 Triton 응답 형식으로 변환합니다."""
        responses = []
        start_idx = 0
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            if input_tensor is None:
                raise ValueError("Input tensor is None")
            num_texts = len(input_tensor.as_numpy())
            end_idx = start_idx + num_texts

            prediction_indices = all_predictions[start_idx:end_idx]
            scores = all_probabilities[start_idx:end_idx]

            label_tensor = pb_utils.Tensor("label", prediction_indices)
            scores_tensor = pb_utils.Tensor("scores", scores)

            inference_response = pb_utils.InferenceResponse(output_tensors=[label_tensor, scores_tensor])
            responses.append(inference_response)
            start_idx = end_idx
        return responses

    def execute(self, requests: list[pb_utils.InferenceRequest]) -> list[pb_utils.InferenceResponse]:
        """`execute` MUST be implemented in every Python model.
        `execute` function receives a list of pb_utils.InferenceRequest as the only argument.
        This function is called when an inference request is made for this model.
        Depending on the batching configuration (e.g. Dynamic Batching) used, `requests` may contain multiple requests.
        Every Python model, must create one pb_utils.InferenceResponse for every pb_utils.InferenceRequest in `requests`.
        If there is an error, you can set the error argument when creating a pb_utils.InferenceResponse

        Args:
            requests (list[pb_utils.InferenceRequest]) : A list of pb_utils.InferenceRequest

        Returns:
            list[pb_utils.InferenceResponse] : A list of pb_utils.InferenceResponse. The length of this list must be the same as `requests`
        """

        # 1. 모든 요청에서 텍스트 파싱
        all_texts = self._parse_requests(requests)
        if not all_texts:
            return []

        # 2. Inference
        outputs = self.model.classify(all_texts)

        # 3. Process
        predictions, probabilities = self._format_outputs(outputs)

        # 4. 최종 응답 생성
        responses = self._create_responses(requests, predictions, probabilities)
        return responses

    def finalize(self) -> None:
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL.
        This function allows the model to perform any necessary clean ups before exit.
        """
        self.logger.log("Cleaning up...", self.logger.INFO)
        del self.model
        del self.tokenizer
