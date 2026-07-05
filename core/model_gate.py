# core/model_gate.py
"""
ModelGate — دروازه مرکزی لود مدل‌های سنگین (MLX / torch)

مشکل: چند مدل بزرگ (phi3 + granite + embedding) همزمان روی Metal لود می‌شوند
و در میانه فلو با «Insufficient Memory» کرش می‌کند.

راه‌حل: همه لودها از این gate عبور می‌کنند. دو حالت قابل کانفیگ:

    "serial"      → قبل از لود هر مدل جدید، مدل‌های غیر-resident قبلی
                    آزاد می‌شوند (فقط یک مدل سنگین همزمان در حافظه).
    "concurrent"  → همه مدل‌ها کش می‌شوند و کنار هم می‌مانند
                    (سریع‌تر، ولی به حافظه بیشتری نیاز دارد).

کانفیگ در configs/main_config.py:

    "model_execution": {
        "mode": "serial",           # یا "concurrent"
        "resident": ["embedding"],  # kind هایی که هرگز خودکار آزاد نمی‌شوند
    }
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _release_accelerator_caches() -> None:
    """آزادسازی کش Metal (MLX) و MPS (torch) — بعد از هر eviction"""
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        else:  # نسخه‌های قدیمی
            mx.metal.clear_cache()
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
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


@dataclass
class _Entry:
    obj: Any
    kind: str
    unloader: Optional[Callable[[], None]] = None
    loaded_at: float = field(default_factory=time.time)
    hits: int = 0


class ModelGate:
    """Singleton — کش و کنترل چرخه حیات مدل‌های سنگین"""

    _instance: Optional["ModelGate"] = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.mode: str = "serial"
        self.resident_kinds: set[str] = {"embedding"}
        self._cache: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    # ── Configuration ─────────────────────────────────────────────────

    def configure(self, config: Optional[dict]) -> None:
        cfg = config or {}
        self.mode = cfg.get("mode", "serial")
        self.resident_kinds = set(cfg.get("resident", ["embedding"]))
        logger.info(
            "🚦 ModelGate configured: mode=%s | resident=%s",
            self.mode, sorted(self.resident_kinds),
        )

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
        در حالت serial قبل از لود، مدل‌های غیر-resident دیگر آزاد می‌شوند.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                entry.hits += 1
                logger.debug("ModelGate cache hit: %s (hits=%d)", key, entry.hits)
                return entry.obj

            if self.mode == "serial" and kind not in self.resident_kinds:
                self._evict_others(keep_key=key)

            before = _metal_active_mb()
            t0 = time.perf_counter()
            obj = loader()
            elapsed = time.perf_counter() - t0
            after = _metal_active_mb()

            self._cache[key] = _Entry(obj=obj, kind=kind, unloader=unloader)

            mem = ""
            if before is not None and after is not None:
                mem = f" | metal: {before:.0f}→{after:.0f}MB"
            logger.info(
                "📦 Model loaded via gate: %s (kind=%s, %.1fs, mode=%s)%s",
                key, kind, elapsed, self.mode, mem,
            )
            return obj

    def register_resident(self, key: str, kind: str, obj: Any,
                          unloader: Optional[Callable[[], None]] = None) -> None:
        """ثبت مدلی که بیرون از gate لود شده (مثلاً embedding) برای دیده‌شدن در آمار"""
        with self._lock:
            if key not in self._cache:
                self._cache[key] = _Entry(obj=obj, kind=kind, unloader=unloader)
                logger.info("📌 Resident model registered: %s (kind=%s)", key, kind)

    def evict(self, key: str) -> None:
        with self._lock:
            entry = self._cache.pop(key, None)
        if entry is None:
            return
        try:
            if entry.unloader:
                entry.unloader()
        except Exception as e:
            logger.warning("Unloader failed for %s: %s", key, e)
        _release_accelerator_caches()
        after = _metal_active_mb()
        logger.info(
            "🗑 Model evicted: %s%s",
            key, f" | metal now {after:.0f}MB" if after is not None else "",
        )

    def evict_all(self, include_resident: bool = False) -> None:
        with self._lock:
            keys = [
                k for k, e in self._cache.items()
                if include_resident or e.kind not in self.resident_kinds
            ]
        for k in keys:
            self.evict(k)

    def _evict_others(self, keep_key: str) -> None:
        """در حالت serial: آزادسازی همه مدل‌های غیر-resident به‌جز keep_key"""
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
            entry = self._cache.pop(k, None)
            if entry and entry.unloader:
                try:
                    entry.unloader()
                except Exception as e:
                    logger.warning("Unloader failed for %s: %s", k, e)
        if victims:
            _release_accelerator_caches()

    # ── Introspection (برای dashboard) ────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            return {
                "mode": self.mode,
                "resident_kinds": sorted(self.resident_kinds),
                "loaded": [
                    {
                        "key": k,
                        "kind": e.kind,
                        "hits": e.hits,
                        "loaded_at": e.loaded_at,
                    }
                    for k, e in self._cache.items()
                ],
                "metal_active_mb": _metal_active_mb(),
            }


def get_model_gate() -> ModelGate:
    return ModelGate()
