from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class FeatureAggregator:
    def compute_sky_ratio(self, bgr: np.ndarray) -> float:
        b = bgr[...,0].astype("float32")
        g = bgr[...,1].astype("float32")
        r = bgr[...,2].astype("float32")
        bright = (0.114*b + 0.587*g + 0.299*r)
        sky = (b > r + 15) & (bright > 120)
        return float(np.mean(sky))

    def build_window_features(self, quality_mean: Dict[str,float], object_histogram: Dict[str,int], frames_used: int, frame_shape_hw: Tuple[int,int]) -> Dict[str,float]:
        h, w = frame_shape_hw
        person_density = float(object_histogram.get("person", 0)) / max(frames_used, 1.0)
        feats: Dict[str,float] = {
            "mean_brightness": float(quality_mean.get("mean_brightness", 0.0)),
            "blur_score": float(quality_mean.get("blur_score", 0.0)),
            "motion_score": float(quality_mean.get("motion_score", 0.0)),
            "occlusion_score": float(quality_mean.get("occlusion_score", 0.0)),
            "person_density": float(person_density),
            "frame_h": float(h),
            "frame_w": float(w),
        }
        for k, v in list(object_histogram.items())[:30]:
            feats[f"obj_{k}"] = float(v) / max(frames_used, 1.0)
        return feats
