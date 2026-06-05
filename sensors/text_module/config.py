from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class RuntimeConfig:
    min_text_length: int = 2
    min_asr_confidence: float = 0.55
    min_command_confidence: float = 0.40

    embedding_enabled: bool = True
    embedding_dim: int = 768
    max_tokens: int = 256

    language_detection_enabled: bool = True
    normalize_whitespace: bool = True
    strip_control_chars: bool = True

    # Optional local datasets used by training/analysis flows.
    dataset_resources: Dict[str, str] = field(default_factory=dict)
