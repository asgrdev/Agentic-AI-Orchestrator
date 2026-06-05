from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class AudioRingBuffer:
    sample_rate_hz: int
    max_seconds: float

    def __post_init__(self) -> None:
        self.capacity_samples = int(self.sample_rate_hz * self.max_seconds)
        self.buffer = np.zeros((self.capacity_samples,), dtype="float32")
        self.write_index = 0
        self.size = 0

    def push(self, pcm: np.ndarray) -> None:
        pcm = pcm.astype("float32")
        n = len(pcm)
        if n >= self.capacity_samples:
            self.buffer[:] = pcm[-self.capacity_samples:]
            self.write_index = 0
            self.size = self.capacity_samples
            return
        end = (self.write_index + n) % self.capacity_samples
        if self.write_index < end:
            self.buffer[self.write_index:end] = pcm
        else:
            first = self.capacity_samples - self.write_index
            self.buffer[self.write_index:] = pcm[:first]
            self.buffer[:end] = pcm[first:]
        self.write_index = end
        self.size = min(self.capacity_samples, self.size + n)

    def read_last(self, n_samples: int) -> np.ndarray:
        n_samples = min(n_samples, self.size)
        start = (self.write_index - n_samples) % self.capacity_samples
        if start < self.write_index:
            return self.buffer[start:self.write_index].copy()
        return np.concatenate([self.buffer[start:], self.buffer[:self.write_index]]).copy()
