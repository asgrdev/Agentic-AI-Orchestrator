from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np
from ..sources.base import AudioChunk
from .ringbuffer import AudioRingBuffer

@dataclass
class RawAudioSegment:
    source_id: str
    t0_ms: int
    t1_ms: int
    sample_rate_hz: int
    pcm: np.ndarray

@dataclass
class StreamSegmenter:
    sample_rate_hz: int = 16000
    segment_duration_sec: float = 1.0
    hop_sec: float = 1.0
    max_buffer_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.segment_samples = int(self.sample_rate_hz * self.segment_duration_sec)
        self.hop_samples = int(self.sample_rate_hz * self.hop_sec)
        self.ring = AudioRingBuffer(self.sample_rate_hz, self.max_buffer_seconds)
        self._samples_since_last_emit = 0

    def push_chunk(self, chunk: AudioChunk) -> Iterator[RawAudioSegment]:
        if chunk.sample_rate_hz != self.sample_rate_hz:
            raise ValueError(f"Segmenter expected {self.sample_rate_hz} Hz but got {chunk.sample_rate_hz} Hz")
        self.ring.push(chunk.pcm)
        self._samples_since_last_emit += len(chunk.pcm)

        while self._samples_since_last_emit >= self.hop_samples and self.ring.size >= self.segment_samples:
            self._samples_since_last_emit -= self.hop_samples
            pcm = self.ring.read_last(self.segment_samples)
            t1_ms = chunk.t_ms
            t0_ms = int(t1_ms - (self.segment_duration_sec * 1000.0))
            yield RawAudioSegment(
                source_id=chunk.source_id,
                t0_ms=t0_ms,
                t1_ms=t1_ms,
                sample_rate_hz=self.sample_rate_hz,
                pcm=pcm,
            )
