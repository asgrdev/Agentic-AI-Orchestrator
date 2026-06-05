from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import queue
import time
import numpy as np
from .base import AudioSource, AudioChunk

@dataclass
class PushAudioSource(AudioSource):
    source_id: str = "push0"
    sample_rate_hz: int = 16000
    max_queue: int = 200

    def __post_init__(self) -> None:
        self._queue: "queue.Queue[AudioChunk]" = queue.Queue(maxsize=self.max_queue)

    def push(self, pcm: np.ndarray, t_ms: Optional[int] = None) -> None:
        if t_ms is None:
            t_ms = int(time.time() * 1000)
        chunk = AudioChunk(
            t_ms=t_ms,
            pcm=pcm.astype("float32"),
            sample_rate_hz=self.sample_rate_hz,
            source_id=self.source_id,
        )
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                pass

    def read(self, timeout: Optional[float] = 0.2) -> Optional[AudioChunk]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        return
