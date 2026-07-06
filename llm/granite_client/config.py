# granite_client/config.py
from dataclasses import dataclass, field
from typing import Optional


def _default_granite_path() -> str:
    # اول models/granite4_3b لوکال پروژه؛ بعد مسیر مطلق قدیمی
    from core.model_paths import resolve_model_path
    return resolve_model_path("granite4_3b", "/Users/dbk/Desktop/RAG/models/granite4_3b")


@dataclass(frozen=True)
class MlxGraniteConfig:
    model_path: str = _default_granite_path()
    max_tokens: int = 1500
    temperature: float = 0.74
    top_p: float = 0.94
    seed: int = 0
    kv_bits: int = 8
    kv_group_size: int = 32
    max_kv_size: int = 4096
    stop: Optional[list] = None
