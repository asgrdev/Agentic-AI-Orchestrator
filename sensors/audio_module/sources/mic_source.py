from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import logging
import platform
import time
from queue import Empty, Full, Queue

import numpy as np

from .base import AudioChunk

try:
    import sounddevice as sd
except Exception:
    sd = None

logger = logging.getLogger(__name__)


@dataclass
class MicAudioSource:
    """Cross-platform microphone source for macOS/Linux/Windows via sounddevice."""

    device: Optional[int] = None
    channels: int = 1
    samplerate: Optional[int] = None
    blocksize: int = 1024
    dtype: str = "float32"
    callback: Optional[Callable[[np.ndarray], None]] = None
    chunk_duration_ms: int = 1000

    _stream: Optional[Any] = field(init=False, default=None)
    _is_running: bool = field(init=False, default=False)
    _chunk_samples: int = field(init=False, default=16000)
    sample_rate_hz: int = field(init=False, default=16000)
    _buffer_queue: Optional[Queue] = field(init=False, default=None)

    def __post_init__(self) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice is required for microphone input. Install: pip install sounddevice"
            )
        self._buffer_queue = Queue(maxsize=8)
        self._initialize_stream()

    @staticmethod
    def list_available_microphones() -> List[int]:
        if sd is None:
            return []
        out: List[int] = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if int(dev.get("max_input_channels", 0)) > 0:
                    out.append(i)
        except Exception:
            return []
        return out

    @staticmethod
    def describe_microphones() -> List[Dict[str, Any]]:
        if sd is None:
            return []
        rows: List[Dict[str, Any]] = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if int(dev.get("max_input_channels", 0)) <= 0:
                    continue
                rows.append(
                    {
                        "index": i,
                        "name": str(dev.get("name", "")),
                        "hostapi": int(dev.get("hostapi", -1)),
                        "max_input_channels": int(dev.get("max_input_channels", 0)),
                        "default_samplerate": float(dev.get("default_samplerate", 0.0)),
                    }
                )
        except Exception:
            return []
        return rows

    def _auto_select_device(self) -> Optional[int]:
        if self.device is not None:
            return int(self.device)
        mics = self.list_available_microphones()
        if not mics:
            return None
        try:
            default_dev = sd.default.device[0] if sd.default.device else None
            if default_dev in mics:
                return int(default_dev)
        except Exception:
            pass
        return int(mics[0])

    def _candidate_samplerates(self, device_index: Optional[int]) -> List[int]:
        common = [16000, 22050, 32000, 44100, 48000, 8000, 11025]
        rates: List[int] = []
        if self.samplerate is not None:
            rates.append(int(self.samplerate))
        if device_index is not None:
            try:
                dev = sd.query_devices(device_index)
                default_sr = int(float(dev.get("default_samplerate", 0)))
                if default_sr > 0:
                    rates.append(default_sr)
            except Exception:
                pass
        rates.extend(common)
        # order-preserving dedup
        out: List[int] = []
        for r in rates:
            if r not in out:
                out.append(r)
        return out

    def _is_supported(self, device_index: Optional[int], samplerate: int) -> bool:
        try:
            sd.check_input_settings(
                device=device_index,
                samplerate=int(samplerate),
                channels=int(self.channels),
                dtype=self.dtype,
            )
            return True
        except Exception:
            return False

    def _pick_samplerate(self, device_index: Optional[int]) -> int:
        for rate in self._candidate_samplerates(device_index):
            if self._is_supported(device_index, rate):
                return int(rate)
        # final fallback: let sounddevice choose default if possible
        if self._is_supported(device_index, 44100):
            return 44100
        raise RuntimeError(
            f"No compatible samplerate found for device={device_index} on {platform.system()}."
        )

    def _initialize_stream(self) -> None:
        device_index = self._auto_select_device()
        if device_index is None:
            raise RuntimeError("No microphone input device found.")

        rate = self._pick_samplerate(device_index)
        self.sample_rate_hz = int(rate)
        self._chunk_samples = max(1, int(self.sample_rate_hz * (self.chunk_duration_ms / 1000.0)))
        self.device = int(device_index)

        logger.info(
            "Initializing microphone stream device=%s samplerate=%s channels=%s",
            self.device,
            self.sample_rate_hz,
            self.channels,
        )

        stream_kwargs = dict(
            device=self.device,
            channels=self.channels,
            samplerate=self.sample_rate_hz,
            blocksize=max(int(self.blocksize), int(self._chunk_samples)),
            dtype=self.dtype,
        )

        # If no callback is needed, keep stream in blocking mode so read() works.
        if self.callback is not None:
            stream_kwargs["callback"] = self._callback

        self._stream = sd.InputStream(**stream_kwargs)
        self.start()

    def _callback(self, indata, frames, ts, status) -> None:
        _ = (frames, ts)
        if status:
            logger.warning("Microphone status: %s", status)
        if self._buffer_queue is not None:
            try:
                self._buffer_queue.put_nowait(indata.copy())
            except Full:
                pass
        if self.callback:
            try:
                self.callback(indata.copy())
            except Exception:
                logger.exception("Microphone callback failed")

    def start(self) -> None:
        if self._stream is not None and not self._is_running:
            self._stream.start()
            self._is_running = True

    def stop(self) -> None:
        if self._stream is not None and self._is_running:
            self._stream.stop()
            self._is_running = False

    def close(self) -> None:
        if self._stream is not None:
            try:
                self.stop()
                self._stream.close()
            finally:
                self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _ = (exc_type, exc_val, exc_tb)
        self.close()

    def read(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        # sounddevice InputStream.read is blocking by frames; timeout kept for API compatibility.
        if self._stream is None:
            return None
        if not self._is_running:
            self.start()

        if self.callback is not None:
            if self._buffer_queue is None:
                return None
            try:
                q_timeout = None if timeout is None else max(0.0, float(timeout))
                data = self._buffer_queue.get(timeout=q_timeout)
            except Empty:
                return None
        else:
            data, _ = self._stream.read(self._chunk_samples)

        pcm = np.asarray(data).reshape(-1).astype("float32")
        return AudioChunk(
            t_ms=int(time.time() * 1000),
            pcm=pcm,
            sample_rate_hz=self.sample_rate_hz,
            source_id="mic",
        )
