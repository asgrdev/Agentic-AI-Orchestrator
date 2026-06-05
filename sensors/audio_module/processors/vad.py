from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class EnergyVad:
    enabled: bool = True
    speech_probability_threshold: float = 0.60
    energy_floor: float = 0.010

    def speech_probability(self, pcm: np.ndarray) -> float:
        if not self.enabled:
            return 0.0
        rms = float(np.sqrt(np.mean(pcm.astype("float32") ** 2) + 1e-12))
        p = (rms - self.energy_floor) / max(self.energy_floor, 1e-6)
        return float(np.clip(p, 0.0, 1.0))

    def infer(self, pcm: np.ndarray) -> tuple[float, bool]:
        p = self.speech_probability(pcm)
        return p, bool(p >= self.speech_probability_threshold)
