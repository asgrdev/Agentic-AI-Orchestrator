from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np

@dataclass
class EmbeddingStore:
    directory: str

    def __post_init__(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    def save(self, seg_id: str, embedding: np.ndarray) -> str:
        path = os.path.join(self.directory, f"{seg_id}.npy")
        np.save(path, embedding.astype("float32"))
        return path
