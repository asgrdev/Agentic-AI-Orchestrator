from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import hashlib
import os
from pathlib import Path
import urllib.request
import warnings
import time
from ..patched_download import install_whisper_download_patch
try:
    import whisper
except Exception:
    whisper = None

try:
    from runtime_module_io_logger import log_module_io
except Exception:  # pragma: no cover
    def log_module_io(*args: Any, **kwargs: Any) -> None:
        return

@dataclass
class WhisperConfig:
    model_name: str = "medium"
    device: str = "cpu"
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 1
    temperature: float = 0.0

class WhisperProcessor:


    def _sanitize_model_name(self,x) -> str:
        # numpy fixed-size bytes/string may contain trailing null bytes
        if isinstance(x, (np.bytes_, bytes, bytearray)):
            x = bytes(x).decode("utf-8", "ignore")
        # numpy str_ etc.
        x = str(x)
    
        # remove embedded null bytes (CRITICAL)
        x = x.replace("\x00", "").strip()
        return x
    def __init__(self, cfg: WhisperConfig) -> None:
        if whisper is None:
            raise RuntimeError("Install whisper: pip install -U openai-whisper")
        self.cfg = cfg
        # Patch whisper downloader (best-effort) but do not load model at import-time.
        install_whisper_download_patch(whisper)

        name = self._sanitize_model_name(cfg.model_name)
        path=self._sanitize_model_name("models/whisper-medium")
        # اگر کاربر پوشه داده مثل models/whisper-medium
        p = Path(name)
        if p.exists() and p.is_dir():
            # فایل medium.pt را انتخاب کن (یا اگر چندتاست، بر اساس cfg)
            cand = p / "medium.pt"
            if cand.exists():
                name = str(cand)
        
        # اگر فایل .pt است، download_root لازم نیست
        if name.endswith(".pt"):
            self.model = whisper.load_model(name, device=cfg.device)
        else:
            # اگر اسم مدل است (مثلاً "medium")، download_root معنی دارد
            self.model = whisper.load_model(name, device=cfg.device, download_root=path)
        
        self._last_run_ms: Optional[int] = None
        log_module_io(
            module="whisper.asr",
            stage="model_load",
            inputs={
                "model_name": str(cfg.model_name),
                "device": str(cfg.device),
                "language": cfg.language,
                "task": cfg.task,
                "beam_size": int(cfg.beam_size),
                "temperature": float(cfg.temperature),
            },
            outputs={"loaded": True, "resolved_model": str(name)},
        )



    def can_run(self, now_ms: int, min_gap_sec: float) -> bool:
        if self._last_run_ms is None:
            return True
        return (now_ms - self._last_run_ms) >= int(min_gap_sec * 1000)

    def transcribe(self, pcm: np.ndarray, sample_rate_hz: int, now_ms: int) -> Dict[str, Any]:
        t0 = time.time()
        log_module_io(
            module="whisper.asr",
            stage="input",
            inputs={
                "sample_rate_hz": int(sample_rate_hz),
                "samples": int(pcm.shape[0]) if hasattr(pcm, "shape") else None,
                "now_ms": int(now_ms),
                "language": self.cfg.language,
                "task": self.cfg.task,
                "temperature": float(self.cfg.temperature),
                "beam_size": int(self.cfg.beam_size),
            },
        )
        if sample_rate_hz != 16000:
            raise ValueError(f"Whisper expects 16000 Hz. Got {sample_rate_hz} Hz.")
        self._last_run_ms = now_ms
        audio = pcm.astype("float32")
        try:
            out = self.model.transcribe(
                audio,
                language=self.cfg.language,
                task=self.cfg.task,
                temperature=self.cfg.temperature,
                beam_size=self.cfg.beam_size,
                fp16=False,
            )
            text = str(out.get("text", "") or "").strip()
            segs = out.get("segments", []) or []
            log_module_io(
                module="whisper.asr",
                stage="output",
                outputs={
                    "text": text,
                    "text_len": len(text),
                    "segments_count": len(segs),
                    "avg_logprob": out.get("avg_logprob"),
                    "no_speech_prob": out.get("no_speech_prob"),
                    "compression_ratio": out.get("compression_ratio"),
                    "latency_ms": round((time.time() - t0) * 1000.0, 3),
                },
            )
            return out
        except Exception as exc:
            log_module_io(
                module="whisper.asr",
                stage="error",
                error=repr(exc),
                metadata={"latency_ms": round((time.time() - t0) * 1000.0, 3)},
            )
            raise
