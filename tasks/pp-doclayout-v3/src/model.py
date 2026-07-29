"""Triton Python backend for PaddlePaddle/PP-DocLayoutV3_safetensors."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import triton_python_backend_utils as pb_utils
from transformers import AutoImageProcessor, AutoModelForObjectDetection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import BBox, LayoutItem, Point


def _to_bytes(raw: Any) -> bytes:
    """Normalize one TYPE_STRING element into raw image bytes."""
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if isinstance(raw, np.ndarray):
        return raw.tobytes()
    if isinstance(raw, str):
        # tritonclient may hand back str when binary_data=False
        return raw.encode("latin-1")
    return bytes(raw)


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        self.logger = pb_utils.Logger

        # Weights are staged next to model.py by scripts/deploy.py
        weight_dir = str(Path(__file__).resolve().parent / "model")

        # Read from config.pbtxt
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        def get_param(name: str, default_value: Any) -> str:
            """Read a `parameters` entry from config.pbtxt"""
            param = params.get(name, {"string_value": str(default_value)})
            return param["string_value"]

        self.logger.log_info(f"Loading {weight_dir} with params {params}")

        self.threshold = float(get_param("DOCLAYOUT_THRESHOLD", "0.5"))
        self.inference_batch_size = int(get_param("INFERENCE_BATCH_SIZE", "8"))

        model_kwargs: dict[str, Any] = dict()
        model_attn_implementation = get_param("MODEL_ATTN_IMPLEMENTATION", "none")
        if model_attn_implementation.lower() != "none":
            model_kwargs["attn_implementation"] = model_attn_implementation

        if get_param("MODEL_USE_FP16", "false").lower() == "true":
            model_kwargs["dtype"] = torch.float16
        elif get_param("MODEL_USE_BF16", "false").lower() == "true":
            model_kwargs["dtype"] = torch.bfloat16

        model_additional_params = json.loads(get_param("MODEL_ADDITIONAL_PARAMS", "{}"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForObjectDetection.from_pretrained(
            weight_dir,
            trust_remote_code=True,
            local_files_only=True,
            **model_kwargs,
            **model_additional_params,
        ).to(self.device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            weight_dir, trust_remote_code=True, local_files_only=True
        )

        # config.json ships string keys, PretrainedConfig may or may not cast them
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

        self.logger.log_info(
            f"Loaded model on {self.device} (threshold={self.threshold}, "
            f"inference_batch_size={self.inference_batch_size})"
        )

    def _parse_request(self, request: pb_utils.InferenceRequest) -> list[Image.Image]:
        """Read images (list[PIL.Image]) from request"""
        input_tensor = pb_utils.get_input_tensor_by_name(request, "image_bytes")
        if input_tensor is None:
            raise ValueError("Input tensor 'image_bytes' is None")
        images: list[Image.Image] = []
        for raw in input_tensor.as_numpy().reshape(-1):
            image = Image.open(io.BytesIO(_to_bytes(raw)))
            images.append(image.convert("RGB"))
        return images

    def _parse_requests(
        self, requests: list[pb_utils.InferenceRequest]
    ) -> tuple[list[Image.Image], list[int], dict[int, str]]:
        """
        Args:
            requests: received batched requests
        Returns:
            list[Image.Image]: flattened input image list
            list[int]: request index for each item
            dict[int, str]: error message per request index that failed to parse
        """
        images: list[Image.Image] = []
        indices: list[int] = []
        errors: dict[int, str] = {}
        for i, request in enumerate(requests):
            # a single undecodable image must not fail the whole batch
            try:
                request_images = self._parse_request(request)
            except Exception as e:
                errors[i] = f"failed to read input: {e}"
                continue
            images.extend(request_images)
            indices.extend([i] * len(request_images))
        return images, indices, errors

    @torch.inference_mode()
    def _infer(self, images: list[Image.Image]) -> list[list[LayoutItem]]:
        """Detect layout items for a batch of pages, ordered by reading order"""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            # target_sizes expects (height, width)
            target_sizes=[(img.height, img.width) for img in images],
            threshold=self.threshold,
        )

        pages: list[list[LayoutItem]] = []
        for result in results:
            order_seq = result.get("order_seq")
            if order_seq is not None:
                sorted_indices = order_seq.argsort().tolist()
            else:
                sorted_indices = list(range(len(result["scores"])))

            polygon_points = result.get("polygon_points")

            items: list[LayoutItem] = []
            for i in sorted_indices:
                score = result["scores"][i].item()
                label_id = result["labels"][i].item()
                x1, y1, x2, y2 = result["boxes"][i].tolist()

                polygon: list[Point] = []
                if polygon_points is not None:
                    polygon = [
                        Point(x=round(float(pt[0])), y=round(float(pt[1])))
                        for pt in polygon_points[i]
                    ]

                items.append(
                    LayoutItem(
                        label=self.id2label.get(label_id, str(label_id)),
                        score=round(score, 4),
                        order=int(order_seq[i].item()) if order_seq is not None else i,
                        bbox=BBox(x1=round(x1), y1=round(y1), x2=round(x2), y2=round(y2)),
                        polygon_points=polygon,
                    )
                )
            pages.append(items)

        return pages

    def _predict(self, images: list[Image.Image]) -> list[list[LayoutItem]]:
        """Detect layout items, chunked by inference_batch_size"""
        pages: list[list[LayoutItem]] = []
        for i in range(0, len(images), self.inference_batch_size):
            pages.extend(self._infer(images[i : i + self.inference_batch_size]))
        return pages

    def execute(self, requests: list[pb_utils.InferenceRequest]) -> list[pb_utils.InferenceResponse]:
        # Parse Requests
        images, indices, errors = self._parse_requests(requests)
        self.logger.log_info(f"Received {len(requests)} requests with {len(images)} images")

        # Inference
        start = time.time()
        try:
            pages = self._predict(images) if images else []
        except Exception as e:
            self.logger.log_error(f"Inference failed: {e}")
            error = pb_utils.TritonError(f"inference failed: {e}")
            return [pb_utils.InferenceResponse(error=error) for _ in requests]
        end = time.time()

        total_items = sum(len(p) for p in pages)
        duration_ms = (end - start) * 1000
        self.logger.log_info(f"Inference complete: {total_items} items in {duration_ms:.2f}ms")

        payloads = np.array(
            [
                json.dumps([item.model_dump() for item in page], ensure_ascii=False).encode("utf-8")
                for page in pages
            ],
            dtype=object,
        )
        request_indices = np.array(indices, dtype=np.int64)

        # Split results back per request
        responses: list[pb_utils.InferenceResponse] = []
        for i in range(len(requests)):
            if i in errors:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(errors[i])))
                continue
            mask = request_indices == i
            # output dims are [ 1 ] under max_batch_size, so keep the trailing axis
            items_tensor = pb_utils.Tensor("layout_items", payloads[mask].reshape(-1, 1))
            responses.append(pb_utils.InferenceResponse(output_tensors=[items_tensor]))

        return responses

    def finalize(self) -> None:
        self.logger.log_info("Finalizing")
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
