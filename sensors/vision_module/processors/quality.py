from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..cv2_compat import cv2

@dataclass
class FrameQualityEstimator:
    def estimate(self, bgr: np.ndarray, previous_bgr: np.ndarray | None) -> dict:
        gray = self._to_gray(bgr)
        mean_brightness = float(np.mean(gray) / 255.0)
        if cv2 is not None:
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0)
        else:
            gx = np.diff(gray.astype('float32'), axis=1)
            blur_score = float(np.var(gx) / 1000.0)
        motion_score = 0.0
        if previous_bgr is not None:
            try:
                prev_gray = self._to_gray(previous_bgr)
                if prev_gray.shape != gray.shape:
                    prev_gray = self._resize_gray(prev_gray, gray.shape)
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    motion_score = float(np.mean(np.abs(gray.astype('float32') - prev_gray.astype('float32'))) / 255.0)
            except Exception:
                motion_score = 0.0
        occlusion_score = float(np.mean(gray < 10))
        return dict(mean_brightness=mean_brightness, blur_score=blur_score, motion_score=motion_score, occlusion_score=occlusion_score)

    @staticmethod
    def _to_gray(bgr: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        b, g, r = bgr[...,0], bgr[...,1], bgr[...,2]
        return (0.114*b + 0.587*g + 0.299*r).astype('uint8')

    @staticmethod
    def _resize_gray(gray: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray | None:
        if gray is None or gray.ndim != 2:
            return None
        dst_h, dst_w = int(target_shape[0]), int(target_shape[1])
        if dst_h <= 0 or dst_w <= 0:
            return None
        src_h, src_w = int(gray.shape[0]), int(gray.shape[1])
        if src_h <= 0 or src_w <= 0:
            return None
        if src_h == dst_h and src_w == dst_w:
            return gray
        if cv2 is not None:
            return cv2.resize(gray, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        y_idx = np.linspace(0, max(src_h - 1, 0), dst_h).astype(np.int64)
        x_idx = np.linspace(0, max(src_w - 1, 0), dst_w).astype(np.int64)
        return gray[np.ix_(y_idx, x_idx)]
