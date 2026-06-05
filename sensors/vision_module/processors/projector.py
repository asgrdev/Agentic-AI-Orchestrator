from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

@dataclass
class ProjectorConfig:
    embedding_dim: int = 512
    hidden_dim: int = 512
    device: str = "cpu"
    model_name: str = "vision_projector_v1"

class VisionProjector:
    def __init__(self, cfg: ProjectorConfig) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for VisionProjector.")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._feature_keys: List[str] = []
        self.model = None

    def _build(self, input_dim: int):
        return nn.Sequential(nn.Linear(input_dim, self.cfg.hidden_dim), nn.ReLU(), nn.Linear(self.cfg.hidden_dim, self.cfg.embedding_dim))

    def _vectorize(self, features: Dict[str,float]) -> np.ndarray:
        if not self._feature_keys:
            self._feature_keys = sorted(features.keys())
        return np.array([float(features.get(k, 0.0)) for k in self._feature_keys], dtype="float32")

    def infer(self, features: Dict[str,float]) -> np.ndarray:
        x = self._vectorize(features)
        if self.model is None:
            self.model = self._build(x.shape[0]).to(self.device).eval()
        xt = torch.from_numpy(x).to(self.device).unsqueeze(0)
        with torch.no_grad():
            z = self.model(xt).squeeze(0).detach().float().cpu().numpy()
        n = float(np.linalg.norm(z) + 1e-12)
        return (z/n).astype("float32")

    @property
    def feature_keys(self) -> List[str]:
        return self._feature_keys
