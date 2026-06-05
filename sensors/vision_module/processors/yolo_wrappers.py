from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np

from ..cv2_compat import CV2_AVAILABLE, CV2_IMPORT_ERROR

if CV2_AVAILABLE:
    try:
        from ultralytics import YOLO
    except Exception:
        YOLO = None
else:
    YOLO = None

try:
    from runtime_module_io_logger import log_module_io
except Exception:  # pragma: no cover
    def log_module_io(*args: Any, **kwargs: Any) -> None:
        return

@dataclass
class YoloConfig:
    model_path: str
    device: str = "cpu"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 200


def _summarize_ultra_result(r0: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        boxes = getattr(r0, "boxes", None)
        out["boxes_count"] = int(len(boxes)) if boxes is not None else 0
    except Exception:
        out["boxes_count"] = 0
    try:
        masks = getattr(r0, "masks", None)
        out["masks_count"] = int(len(masks)) if masks is not None else 0
    except Exception:
        out["masks_count"] = 0
    try:
        kpts = getattr(r0, "keypoints", None)
        out["keypoints_present"] = bool(kpts is not None)
        out["keypoints_shape"] = list(kpts.data.shape) if (kpts is not None and hasattr(kpts, "data")) else None
    except Exception:
        out["keypoints_present"] = False
        out["keypoints_shape"] = None
    return out

class YoloDetector:
    def __init__(self, cfg: YoloConfig) -> None:
        if YOLO is None:
            if not CV2_AVAILABLE:
                raise RuntimeError(
                    "ultralytics/cv2 unavailable due to OpenCV compatibility issue "
                    f"({CV2_IMPORT_ERROR}). Pin numpy<2 and reinstall requirements."
                )
            raise RuntimeError("ultralytics is required. Install: pip install ultralytics")
        self.cfg = cfg
        self.model = YOLO(cfg.model_path )
        log_module_io(
            module="yolo.detector",
            stage="model_load",
            inputs={"model_path": cfg.model_path, "device": cfg.device},
            outputs={"loaded": True},
        )

    def infer(self, bgr: np.ndarray) -> List[tuple[str, float, Tuple[float,float,float,float]]]:
        t0 = time.time()
        log_module_io(
            module="yolo.detector",
            stage="input",
            inputs={
                "frame_shape": list(bgr.shape),
                "frame_dtype": str(bgr.dtype),
                "conf_threshold": float(self.cfg.conf_threshold),
                "iou_threshold": float(self.cfg.iou_threshold),
                "max_det": int(self.cfg.max_det),
                "device": self.cfg.device,
            },
        )
        try:
            res = self.model.predict(
                source=bgr,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                max_det=self.cfg.max_det,
                device=self.cfg.device,
                verbose=False,
            )
            r0 = res[0]
            out = []
            if getattr(r0, "boxes", None) is None:
                log_module_io(
                    module="yolo.detector",
                    stage="output",
                    outputs={"detections_count": 0, "latency_ms": round((time.time() - t0) * 1000.0, 3)},
                )
                return out
            names = r0.names
            boxes = r0.boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for bb, c, k in zip(xyxy, conf, cls):
                out.append((names.get(int(k), f"cls_{k}"), float(c), (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))))
            log_module_io(
                module="yolo.detector",
                stage="output",
                outputs={
                    "detections_count": len(out),
                    "detections_top": [
                        {"label": str(lbl), "conf": float(score), "bbox_xyxy": [float(x) for x in bb]}
                        for (lbl, score, bb) in out[:8]
                    ],
                    "result_summary": _summarize_ultra_result(r0),
                    "latency_ms": round((time.time() - t0) * 1000.0, 3),
                },
            )
            return out
        except Exception as exc:
            log_module_io(
                module="yolo.detector",
                stage="error",
                error=repr(exc),
                metadata={"latency_ms": round((time.time() - t0) * 1000.0, 3)},
            )
            raise

class YoloSegmenter:
    def __init__(self, cfg: YoloConfig) -> None:
        if YOLO is None:
            if not CV2_AVAILABLE:
                raise RuntimeError(
                    "ultralytics/cv2 unavailable due to OpenCV compatibility issue "
                    f"({CV2_IMPORT_ERROR}). Pin numpy<2 and reinstall requirements."
                )
            raise RuntimeError("ultralytics is required. Install: pip install ultralytics")
        self.cfg = cfg
        self.model = YOLO(cfg.model_path )
        log_module_io(
            module="yolo.segmenter",
            stage="model_load",
            inputs={"model_path": cfg.model_path, "device": cfg.device},
            outputs={"loaded": True},
        )
    def infer(self, bgr: np.ndarray):
        t0 = time.time()
        log_module_io(
            module="yolo.segmenter",
            stage="input",
            inputs={
                "frame_shape": list(bgr.shape),
                "frame_dtype": str(bgr.dtype),
                "conf_threshold": float(self.cfg.conf_threshold),
                "iou_threshold": float(self.cfg.iou_threshold),
                "max_det": int(self.cfg.max_det),
                "device": self.cfg.device,
            },
        )
        try:
            r0 = self.model.predict(
                source=bgr,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                max_det=self.cfg.max_det,
                device=self.cfg.device,
                verbose=False,
            )[0]
            log_module_io(
                module="yolo.segmenter",
                stage="output",
                outputs={
                    "result_summary": _summarize_ultra_result(r0),
                    "latency_ms": round((time.time() - t0) * 1000.0, 3),
                },
            )
            return r0
        except Exception as exc:
            log_module_io(
                module="yolo.segmenter",
                stage="error",
                error=repr(exc),
                metadata={"latency_ms": round((time.time() - t0) * 1000.0, 3)},
            )
            raise

class YoloPose:
    def __init__(self, cfg: YoloConfig) -> None:
        if YOLO is None:
            if not CV2_AVAILABLE:
                raise RuntimeError(
                    "ultralytics/cv2 unavailable due to OpenCV compatibility issue "
                    f"({CV2_IMPORT_ERROR}). Pin numpy<2 and reinstall requirements."
                )
            raise RuntimeError("ultralytics is required. Install: pip install ultralytics")
        self.cfg = cfg
        self.model = YOLO(cfg.model_path  )
        log_module_io(
            module="yolo.pose",
            stage="model_load",
            inputs={"model_path": cfg.model_path, "device": cfg.device},
            outputs={"loaded": True},
        )
    def infer(self, bgr: np.ndarray):
        t0 = time.time()
        log_module_io(
            module="yolo.pose",
            stage="input",
            inputs={
                "frame_shape": list(bgr.shape),
                "frame_dtype": str(bgr.dtype),
                "conf_threshold": float(self.cfg.conf_threshold),
                "iou_threshold": float(self.cfg.iou_threshold),
                "max_det": int(self.cfg.max_det),
                "device": self.cfg.device,
            },
        )
        try:
            r0 = self.model.predict(
                source=bgr,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                max_det=self.cfg.max_det,
                device=self.cfg.device,
                verbose=False,
            )[0]
            log_module_io(
                module="yolo.pose",
                stage="output",
                outputs={
                    "result_summary": _summarize_ultra_result(r0),
                    "latency_ms": round((time.time() - t0) * 1000.0, 3),
                },
            )
            return r0
        except Exception as exc:
            log_module_io(
                module="yolo.pose",
                stage="error",
                error=repr(exc),
                metadata={"latency_ms": round((time.time() - t0) * 1000.0, 3)},
            )
            raise
