# granite_client/__init__.py

from .config import MlxGraniteConfig
from .base_client import BaseGraniteClient, LLMResponse
from .mlx_client import MlxGraniteClient
from .prompts import (
    SYSTEM_DEFAULT,
    SYSTEM_REASONING,
    SYSTEM_RELATION,
    SYSTEM_CITATION,
    build_prompt,
)

__all__ = [
    "MlxGraniteConfig",
    "BaseGraniteClient",
    "LLMResponse",
    "MlxGraniteClient",
    "SYSTEM_DEFAULT",
    "SYSTEM_REASONING",
    "SYSTEM_RELATION",
    "SYSTEM_CITATION",
    "build_prompt",
]
