"""
src/detection/detector.py — YOLO-based object detector.
Designed for RTX 3050 4GB: uses small model, FP16, single-frame batches.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import cfg

logger = logging.getLogger(__name__)

# COCO class names we care about
PRIORITY_CLASSES = {
    "person", "car", "bicycle", "motorcycle",
    "bus", "truck", "backpack", "handbag", "suitcase",
}

# COCO ID → name mapping (subset)
COCO_NAMES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 24: "backpack", 26: "handbag", 28: "suitcase",
}


class Detection:
    """Single detection result."""
    __slots__ = ["x1", "y1", "x2", "y2", "confidence", "class_id", "class_name"]

    def __init__(self, x1: int, y1: int, x2: int, y2: int,
                 confidence: float, class_id: int, class_name: str):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def __repr__(self) -> str:
        return (f"Detection({self.class_name}, conf={self.confidence:.2f}, "
                f"bbox=({self.x1},{self.y1},{self.x2},{self.y2}))")


class YOLODetector:
    """
    Lightweight YOLO object detector.

    Loads model once, runs FP16 inference on GPU if available.
    Filters to priority classes only.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence: float = None,
                 image_size: int = None,
                 device: str = None):
        self.model_path = model_path or cfg.yolo_model
        self.confidence = confidence or cfg.confidence_threshold
        self.image_size = image_size or cfg.image_size
        self.device = device or cfg.device
        self._model = None
        self._available_classes: Dict[int, str] = {}
        self._loaded = False

    def load(self) -> None:
        """Load YOLO model (call once before processing)."""
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLO model: %s on %s", self.model_path, self.device)
            self._model = YOLO(self.model_path)

            # Enable FP16 if on CUDA
            if self.device.startswith("cuda") and cfg.use_fp16:
                try:
                    self._model.model.half()
                    logger.info("YOLO: FP16 enabled")
                except Exception as e:
                    logger.warning("FP16 not available: %s", e)

            # Map available class IDs
            names = self._model.names  # {id: name}
            for cls_id, cls_name in names.items():
                if cls_name in PRIORITY_CLASSES or cls_name in cfg.target_classes:
                    self._available_classes[cls_id] = cls_name

            logger.info("YOLO loaded. Available priority classes: %s",
                        list(self._available_classes.values()))
            self._loaded = True

        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            raise

    def unload(self) -> None:
        """Release model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("YOLO model unloaded from memory")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Returns list of Detection objects filtered to priority classes.
        """
        if not self._loaded:
            self.load()

        try:
            results = self._model(
                frame,
                imgsz=self.image_size,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )

            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id not in self._available_classes:
                        continue
                    conf = float(boxes.conf[i].item())
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    detections.append(Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self._available_classes[cls_id],
                    ))
            return detections

        except Exception as e:
            logger.error("Detection error: %s", e)
            return []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()


# Alias for convenience
ObjectDetector = YOLODetector
