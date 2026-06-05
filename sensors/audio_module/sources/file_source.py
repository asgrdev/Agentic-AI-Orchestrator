from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
from .base import AudioSource, AudioChunk

try:
    import soundfile as sf
except Exception:
    sf = None

@dataclass
class FileAudioSource(AudioSource):
    file_path: str
    source_id: str = "file"
    chunk_duration_ms: int = 40
    target_sample_rate_hz: int = 16000
    realtime: bool = False

    def __post_init__(self) -> None:
        if sf is None:
            raise RuntimeError("soundfile is required. Install: pip install soundfile")
        self._file = sf.SoundFile(self.file_path, mode="r")
        self._sr = int(self._file.samplerate)
        self._chunk_samples_native = int(self._sr * (self.chunk_duration_ms / 1000.0))
        self._read_samples_total = 0
        self._started = time.monotonic()

    def read(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        # timeout is accepted for API compatibility with live polling callers.
        _ = timeout
        frames = self._file.read(self._chunk_samples_native, dtype="float32", always_2d=True)
        if frames.size == 0:
            return None
        self._read_samples_total += frames.shape[0]
        pcm = frames.mean(axis=1)
        pcm = self._resample_if_needed(pcm, self._sr, self.target_sample_rate_hz)
        if self.realtime:
            expected = self._read_samples_total / float(self._sr)
            elapsed = time.monotonic() - self._started
            if expected > elapsed:
                time.sleep(expected - elapsed)
        return AudioChunk(
            t_ms=int(time.time() * 1000),
            pcm=pcm.astype("float32"),
            sample_rate_hz=self.target_sample_rate_hz,
            source_id=self.source_id,
        )

    def close(self) -> None:
        try:
            self._file.close()
        except Exception as e:
          print(f">>{e}")
          pass

    @staticmethod
    def _resample_if_needed(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return pcm
        ratio = dst_sr / float(src_sr)
        x_old = np.arange(len(pcm))
        x_new = np.linspace(0, len(pcm) - 1, int(len(pcm) * ratio))
        return np.interp(x_new, x_old, pcm).astype("float32")
