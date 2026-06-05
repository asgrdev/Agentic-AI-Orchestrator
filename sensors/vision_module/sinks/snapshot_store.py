from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
from ..cv2_compat import cv2

@dataclass
class SnapshotStore:
    directory: str
    jpeg_quality: int = 90
    def __post_init__(self) -> None:
        os.makedirs(self.directory, exist_ok=True)
        if cv2 is None:
            raise RuntimeError("opencv-python is required for SnapshotStore.")
    def save(self, seg_id: str, frame_id: int, bgr: np.ndarray) -> str:
        path = os.path.join(self.directory, f"{seg_id}_frame{frame_id}.jpg")
        ok = cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
        if not ok:
            raise RuntimeError(f"Failed to write snapshot: {path}")
        return path
