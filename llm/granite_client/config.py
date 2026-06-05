# granite_client/config.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MlxGraniteConfig:
    model_path: str = "/Users/dbk/Desktop/RAG/models/granite4-7b"#"granite4-7b-instruct"
    max_tokens: int = 1500
    temperature: float = 0.74
    top_p: float = 0.94
    seed: int = 0
    kv_bits: int = 8
    kv_group_size: int = 32
    max_kv_size: int = 4096
    stop: Optional[list] = None
