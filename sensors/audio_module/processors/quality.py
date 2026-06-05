from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import math

@dataclass
class AudioQualityEstimator:
    clip_threshold: float = 0.98

    def estimate(self, pcm: np.ndarray) -> dict:
        pcm = pcm.astype("float32")
        rms = float(np.sqrt(np.mean(pcm ** 2) + 1e-12))
        dc_offset = float(np.mean(pcm))
        clip_ratio = float(np.mean(np.abs(pcm) >= self.clip_threshold))
        mad = float(np.median(np.abs(pcm - np.median(pcm))) + 1e-6)
        snr_est = float(20.0 * math.log10((rms + 1e-6) / mad))
        return {"rms": rms, "snr_est": snr_est, "clip_ratio": clip_ratio, "dc_offset": dc_offset}
