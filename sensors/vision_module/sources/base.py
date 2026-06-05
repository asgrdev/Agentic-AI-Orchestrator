from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol
import numpy as np

@dataclass
class VideoFrame:
    frame_id: int
    t_ms: int
    bgr: np.ndarray
    source_id: str
    metadata: Optional[Dict[str, Any]] = None

class VideoSource(Protocol):
    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]: ...
    def close(self) -> None: ...
