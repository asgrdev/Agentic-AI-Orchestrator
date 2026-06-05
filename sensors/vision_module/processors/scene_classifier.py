from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class SceneClassifier:
    def infer(self, features: Dict[str,float]) -> Tuple[str, str|None, float]:
        brightness = features.get("mean_brightness", 0.0)
        motion = features.get("motion_score", 0.0)
        person_density = features.get("person_density", 0.0)
        sky_ratio = features.get("sky_ratio", 0.0)

        outdoor = 1.2*sky_ratio + 0.6*max(0.0, brightness-0.45) + 0.2*motion - 0.2*person_density
        indoor = 0.4*person_density + 0.3*max(0.0, 0.45-brightness)

        if outdoor >= indoor:
            label1 = "outdoor"
            label2 = "nature" if sky_ratio > 0.25 else "street"
            conf = float(np.clip(0.55 + (outdoor-indoor), 0.0, 1.0))
            return label1, label2, conf
        label1 = "indoor"
        label2 = "office" if person_density > 0.12 else "home"
        conf = float(np.clip(0.55 + (indoor-outdoor), 0.0, 1.0))
        return label1, label2, conf
