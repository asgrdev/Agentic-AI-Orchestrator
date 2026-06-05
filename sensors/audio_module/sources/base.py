from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

@dataclass
class AudioChunk:
    t_ms: int
    pcm: np.ndarray  # float32 mono [-1,1]
    sample_rate_hz: int
    source_id: str

class AudioSource(Protocol):
    def read(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        ...

    def close(self) -> None:
        ...
