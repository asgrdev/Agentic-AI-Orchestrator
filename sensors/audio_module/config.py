from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime-tunable knobs for speed/accuracy trade-offs."""
    sample_rate_hz: int = 16000

    # Segmentation
    segment_duration_sec: float = 1.0
    segment_hop_sec: float = 1.0  # 1.0 => non-overlapping
    max_buffer_seconds: float = 30.0

    # EfficientAT cadence
    tagging_every_sec: float = 1.0
    embedding_every_sec: float = 1.0

    # Whisper cadence (expensive)
    asr_min_gap_sec: float = 2.0
    asr_max_queue: int = 2

    # Gating
    vad_speech_prob_min: float = 0.60
    tag_speech_prob_min: float = 0.50
    max_noise_tag_prob: float = 0.75

    # Output
    topk_tags: int = 8

    # Optional local datasets used by training/analysis flows.
    dataset_resources: Dict[str, str] = field(default_factory=dict)
