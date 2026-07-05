# core/model_gate.py
"""
ModelGate — دروازه مرکزی لود و اجرای مدل‌های سنگین (MLX / torch)

مشکل: چند مدل بزرگ (phi3 + granite + embedding) همزمان روی Metal لود می‌شوند
و در میانه فلو با «[METAL] Insufficient Memory» کرش می‌کند.

راه‌حل: همه لودها از این gate عبور می‌کنند. دو حالت قابل کانفیگ:

    "serial"      → قبل از لود هر مدل جدید، مدل‌های غیر-resident قبلی
                    آزاد می‌شوند (فقط یک مدل سنگین همزمان در حافظه) و
                    inference ها هم پشت یک قفل سراسری یکی‌یکی اجرا می‌شوند.
    "concurrent"  → همه مدل‌ها کش می‌شوند و کنار هم می‌مانند؛ inference
                    موازی با سقف max_concurrent_inferences
                    (سریع‌تر، ولی به حافظه بیشتری نیاز دارد).

کانفیگ در configs/main_config.py (متغیر محیطی MODEL_EXECUTION_MODE اولویت دارد):

    "model_execution": {
        "mode": "serial",                # یا "concurrent"
        "resident": ["embedding"],       # kind هایی که هرگز خودکار آزاد نمی‌شوند
        "max_concurrent_inferences": 2,  # فقط در حالت concurrent
        "log_memory": True,              # لاگ RSS/Metal دور هر لود/آزادسازی
    }

استفاده در کلاینت‌ها:

    gate = get_model_gate()
    with gate.execution_slot("mlx:/path"):
        model, tokenizer = gate.acquire(key="mlx:/path", kind="llm", loader=...)
        out = generate(model, tokenizer, ...)

نکته مهم: کلاینت‌ها نباید مدل را در attribute نگه دارند؛ در هر فراخوانی
acquire کنند (برای مدل رزیدنت فقط یک dict-lookup است) تا بعد از evict شدن،
لود مجدد به‌صورت خودکار انجام شود.
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

ExecutionMode = Literal["serial", "concurrent"]

_VALID_MODES = ("serial", "concurrent")


def _release_accelerator_caches() -> None:
    """آزادسازی کش Metal (MLX) و CUDA/MPS (torch) — بعد از هر eviction"""
    gc.collect()
    try:
        import mlx.core as mx
        # mx.metal.clear_cache در نسخه‌های جدید mlx deprecated است
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:  # نسخه‌های قدیمی
            mx.metal.clear_cache()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _metal_active_mb() -> Optional[float]:
    """حافظه فعال Metal به MB (اگر MLX موجود باشد)"""
    try:
        import mlx.core as mx
        return mx.get_active_memory() / 1024 ** 2
    except Exception:
        return None


def _process_rss_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except Exception:
        return None


def _memory_suffix() -> str:
    """پسوند لاگ با وضعیت فعلی حافظه (RSS پروسه + Metal اگر موجود باشد)"""
    parts = []
    rss = _process_rss_mb()
    if rss is not None:
        parts.append(f"rss={rss:.0f}MB")
    metal = _metal_active_mb()
    if metal is not None:
        parts.append(f"metal={metal:.0f}MB")
    return f" | {' '.join(parts)}" if parts else ""


@dataclass
class _Entry:
    obj: Any
    kind: str                                   # "llm" | "embedding" | ...
    unloader: Optional[Callable[[], None]] = None
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hits: int = 0
    load_seconds: float = 0.0


class ModelGate:
    """کش + سیاست serial/concurrent برای مدل‌های سنگین (thread-safe)"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: dict[str, _Entry] = {}
        self._mode: ExecutionMode = self._mode_from_env()
        self.resident_kinds: set[str] = {"embedding"}
        self._log_memory: bool = True
        # serial: قفل سراسری → یک inference همزمان
        # (RLock تا acquire داخل execution_slot در همان thread گیر نکند)
        self._serial_slot = threading.RLock()
        # concurrent: سقف inference های موازی
        self._max_concurrent = 2
        self._concurrent_slot = threading.BoundedSemaphore(self._max_concurrent)
        logger.info("🚦 ModelGate ready | mode=%s", self._mode)

    # ── Configuration ─────────────────────────────────────────────────

    @staticmethod
    def _mode_from_env() -> ExecutionMode:
        mode = os.getenv("MODEL_EXECUTION_MODE", "serial").strip().lower()
        if mode not in _VALID_MODES:
            logger.warning(
                "Unknown MODEL_EXECUTION_MODE=%r — falling back to 'serial'", mode
            )
            return "serial"
        return mode  # type: ignore[return-value]

    def configure(self, config: Optional[dict]) -> None:
        """اعمال بلاک "model_execution" از CONFIG (env همیشه اولویت دارد)"""
        cfg = config or {}
        with self._lock:
            mode = (os.getenv("MODEL_EXECUTION_MODE") or cfg.get("mode") or self._mode)
            mode = mode.strip().lower()
            if mode not in _VALID_MODES:
                logger.warning("Invalid model execution mode %r ignored", mode)
            elif mode != self._mode:
                self._mode = mode  # type: ignore[assignment]
                if mode == "serial" and len(self._cache) > 1:
                    # با ورود به حالت serial فقط جدیدترین مدلِ غیر-resident بماند
                    self._evict_others_locked(keep_key=self._most_recent_key())

            if "resident" in cfg:
                self.resident_kinds = set(cfg["resident"])

            max_conc = cfg.get("max_concurrent_inferences")
            if max_conc and int(max_conc) > 0:
                self._max_concurrent = int(max_conc)
                self._concurrent_slot = threading.BoundedSemaphore(self._max_concurrent)

            if "log_memory" in cfg:
                self._log_memory = bool(cfg["log_memory"])

        logger.info(
            "🚦 ModelGate configured: mode=%s | resident=%s | max_concurrent_inferences=%d",
            self._mode, sorted(self.resident_kinds), self._max_concurrent,
        )

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    # ── Main API ──────────────────────────────────────────────────────

    def acquire(
        self,
        key: str,
        kind: str,
        loader: Callable[[], Any],
        unloader: Optional[Callable[[], None]] = None,
    ) -> Any:
        """
        دریافت مدل — از کش اگر موجود باشد، وگرنه لود.

        در حالت serial قبل از لودِ مدلِ جدید، مدل‌های غیر-resident دیگر آزاد
        می‌شوند و برای لود صبر می‌شود تا inference در حال اجرا تمام شود —
        وگرنه مدل جدید وسط generate مدل قبلی لود می‌شود و هر دو موقتاً روی
        GPU می‌مانند (همان OOM قدیمی).
        """
        # fast path: مدل رزیدنت است
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.hits += 1
                entry.last_used = time.time()
                return entry.obj

        if self._mode == "serial":
            # ترتیب گرفتن قفل‌ها همیشه slot → lock است (execution_slot هم
            # RLock است)، پس acquire داخل execution_slot deadlock نمی‌شود
            with self._serial_slot:
                with self._lock:
                    entry = self._cache.get(key)  # double-check بعد از انتظار
                    if entry is not None:
                        entry.hits += 1
                        entry.last_used = time.time()
                        return entry.obj
                    if kind not in self.resident_kinds:
                        self._evict_others_locked(keep_key=key)
                    return self._load_locked(key, kind, loader, unloader)

        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.hits += 1
                entry.last_used = time.time()
                return entry.obj
            return self._load_locked(key, kind, loader, unloader)

    def register_resident(self, key: str, kind: str, obj: Any,
                          unloader: Optional[Callable[[], None]] = None) -> None:
        """ثبت مدلی که بیرون از gate لود شده برای دیده‌شدن در آمار/eviction"""
        with self._lock:
            if key not in self._cache:
                self._cache[key] = _Entry(obj=obj, kind=kind, unloader=unloader)
                logger.info("📌 Resident model registered: %s (kind=%s)", key, kind)

    def evict(self, key: str) -> bool:
        """آزادسازی یک مدل مشخص"""
        with self._lock:
            return self._evict_locked(key)

    def evict_all(self, include_resident: bool = False) -> None:
        """آزادسازی مدل‌ها (پیش‌فرض: به‌جز resident ها)"""
        with self._lock:
            keys = [
                k for k, e in self._cache.items()
                if include_resident or e.kind not in self.resident_kinds
            ]
            for k in keys:
                self._evict_locked(k)

    # ── Execution slot (serial/concurrent inference) ──────────────────

    @contextmanager
    def execution_slot(self, key: str = ""):
        """
        جای اجرای inference:
        - serial:     قفل سراسری — همزمان فقط یک generate/embed اجرا می‌شود
        - concurrent: سمافور با سقف max_concurrent_inferences
        """
        if self._mode == "serial":
            with self._serial_slot:
                yield
        else:
            with self._concurrent_slot:
                yield

    # ── Introspection (برای dashboard) ────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "mode": self._mode,
                "resident_kinds": sorted(self.resident_kinds),
                "max_concurrent_inferences": self._max_concurrent,
                "loaded": [
                    {
                        "key": k,
                        "kind": e.kind,
                        "hits": e.hits,
                        "loaded_at": e.loaded_at,
                        "last_load_seconds": round(e.load_seconds, 2),
                        "idle_seconds": round(time.time() - e.last_used, 1),
                    }
                    for k, e in self._cache.items()
                ],
                "metal_active_mb": _metal_active_mb(),
                "process_rss_mb": _process_rss_mb(),
            }

    # ── Internals (همه زیر self._lock صدا زده می‌شوند) ────────────────

    def _most_recent_key(self) -> Optional[str]:
        candidates = {
            k: e for k, e in self._cache.items()
            if e.kind not in self.resident_kinds
        }
        if not candidates:
            return None
        return max(candidates.items(), key=lambda kv: kv[1].last_used)[0]

    def _evict_others_locked(self, keep_key: Optional[str]) -> None:
        victims = [
            k for k, e in self._cache.items()
            if k != keep_key and e.kind not in self.resident_kinds
        ]
        if victims:
            logger.info(
                "🚦 Serial mode: evicting %d model(s) before loading '%s': %s",
                len(victims), keep_key, victims,
            )
        for k in victims:
            self._evict_locked(k)

    def _evict_locked(self, key: str) -> bool:
        entry = self._cache.pop(key, None)
        if entry is None:
            return False
        if entry.unloader is not None:
            try:
                entry.unloader()
            except Exception as e:
                logger.warning("Unloader failed for %s: %s", key, e)
        del entry
        _release_accelerator_caches()
        logger.info(
            "🗑 Model evicted: %s%s",
            key, _memory_suffix() if self._log_memory else "",
        )
        return True

    def _load_locked(
        self,
        key: str,
        kind: str,
        loader: Callable[[], Any],
        unloader: Optional[Callable[[], None]],
    ) -> Any:
        before = _memory_suffix() if self._log_memory else ""
        logger.info("📥 [gate] loading %s model | key=%s | mode=%s%s",
                    kind, key, self._mode, before)

        t0 = time.perf_counter()
        obj = loader()
        elapsed = time.perf_counter() - t0

        self._cache[key] = _Entry(
            obj=obj, kind=kind, unloader=unloader, load_seconds=elapsed
        )
        logger.info(
            "📦 Model loaded via gate: %s (kind=%s, %.1fs, mode=%s)%s | resident=%s",
            key, kind, elapsed, self._mode,
            _memory_suffix() if self._log_memory else "",
            list(self._cache),
        )
        return obj


# ── Singleton ──────────────────────────────────────────────────────────

_gate: Optional[ModelGate] = None
_gate_lock = threading.Lock()


def get_model_gate() -> ModelGate:
    """دریافت نمونه سراسری ModelGate"""
    global _gate
    if _gate is None:
        with _gate_lock:
            if _gate is None:
                _gate = ModelGate()
    return _gate


def configure_model_gate(config: Optional[dict]) -> ModelGate:
    """اعمال بلاک "model_execution" از CONFIG روی gate سراسری"""
    gate = get_model_gate()
    gate.configure(config)
    return gate
