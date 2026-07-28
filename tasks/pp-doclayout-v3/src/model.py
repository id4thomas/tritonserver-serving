"""Triton Python backend for PaddlePaddle/PP-DocLayoutV3_safetensors."""
import io
import json
import logging
import os
import sys
import time

import numpy as np
import triton_python_backend_utils as pb_utils

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import BBox, LayoutItem, Point


MODEL_PATH = os.environ.get("DOCLAYOUT_MODEL_PATH", "/tmp/model")
THRESHOLD = float(os.environ.get("DOCLAYOUT_THRESHOLD", "0.5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [doclayout] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("doclayout")


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = THRESHOLD

        logger.info(
            "initialize: model_path=%s device=%s threshold=%s",
            MODEL_PATH, self.device, self.threshold,
        )

        self.model = AutoModelForObjectDetection.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        logger.info("model + processor loaded")

    @torch.inference_mode()
    def execute(self, requests):
        all_images: list[Image.Image] = []
        request_counts: list[int] = []

        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "image_bytes")
            arr = in_tensor.as_numpy()
            imgs: list[Image.Image] = []
            for raw in arr.reshape(-1):
                if isinstance(raw, (bytes, bytearray)):
                    buf = bytes(raw)
                elif isinstance(raw, np.ndarray):
                    buf = raw.tobytes()
                elif isinstance(raw, str):
                    buf = raw.encode("latin-1")
                else:
                    buf = bytes(raw)
                imgs.append(Image.open(io.BytesIO(buf)).convert("RGB"))
            request_counts.append(len(imgs))
            all_images.extend(imgs)

        logger.info(
            "execute: requests=%d images=%d per_request=%s",
            len(requests), len(all_images), request_counts,
        )

        t0 = time.perf_counter()
        page_results = self._infer(all_images) if all_images else []
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_items = sum(len(p) for p in page_results)
        logger.info(
            "inference: images=%d items=%d elapsed_ms=%.1f",
            len(all_images), total_items, elapsed_ms,
        )

        responses = []
        idx = 0
        for count in request_counts:
            pages = page_results[idx:idx + count]
            idx += count
            payloads = [
                json.dumps([item.model_dump() for item in page]).encode("utf-8")
                for page in pages
            ]
            out = np.array(payloads, dtype=object).reshape(count, 1)
            out_tensor = pb_utils.Tensor("layout_items", out)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def _infer(self, images: list[Image.Image]) -> list[list[LayoutItem]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[img.size[::-1] for img in images],
            threshold=self.threshold,
        )

        all_pages: list[list[LayoutItem]] = []
        for result in results:
            order_seq = result.get("order_seq")
            if order_seq is not None:
                sorted_indices = order_seq.argsort().tolist()
            else:
                sorted_indices = list(range(len(result["scores"])))

            items: list[LayoutItem] = []
            for i in sorted_indices:
                score = result["scores"][i].item()
                label_id = result["labels"][i].item()
                label_name = self.model.config.id2label[label_id]
                x1, y1, x2, y2 = result["boxes"][i].tolist()

                polygon: list[Point] = []
                if "polygon_points" in result:
                    for pt in result["polygon_points"][i]:
                        polygon.append(Point(x=int(pt[0]), y=int(pt[1])))

                items.append(
                    LayoutItem(
                        label=label_name,
                        score=round(score, 4),
                        order=int(order_seq[i].item()) if order_seq is not None else i,
                        bbox=BBox(x1=round(x1), y1=round(y1), x2=round(x2), y2=round(y2)),
                        polygon_points=polygon,
                    )
                )
            all_pages.append(items)

        return all_pages

    def finalize(self):
        del self.model
        del self.processor
