from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AudioQuality:
    rms: float
    snr_est: float
    clip_ratio: float
    dc_offset: float

@dataclass
class VadResult:
    enabled: bool
    speech_probability: float
    speech_detected: bool
    threshold: float

@dataclass
class TagItem:
    label: str
    probability: float

@dataclass
class TaggingResult:
    model_name: str
    enabled: bool
    topk: List[TagItem]
    tag_entropy: float
    confidence: float

@dataclass
class EmbeddingResult:
    model_name: str
    enabled: bool
    dimension: int
    vector_path: Optional[str] = None
    l2_norm: Optional[float] = None

@dataclass
class AsrWord:
    word: str
    start_sec: float
    end_sec: float
    probability: Optional[float] = None

@dataclass
class AsrResult:
    model_name: str
    enabled: bool
    ran: bool
    run_reason: str
    language: Optional[str] = None
    text: str = ""
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    temperature: Optional[float] = None
    words: List[AsrWord] = field(default_factory=list)

@dataclass
class AudioSegPolicy:
    tagging_every_sec: float
    embedding_every_sec: float
    asr_min_gap_sec: float
    vad_speech_prob_min: float
    tag_speech_prob_min: float
    max_noise_tag_prob: float

@dataclass
class AudioSegConfig:
    sample_rate_hz: int
    topk_tags: int
    policy: AudioSegPolicy

@dataclass
class AudioSeg:
    seg_id: str
    source_id: str
    t0_ms: int
    t1_ms: int
    sample_rate_hz: int
    n_samples: int

    quality: AudioQuality
    vad: VadResult
    tags: Optional[TaggingResult] = None
    embedding: Optional[EmbeddingResult] = None
    asr: Optional[AsrResult] = None

    aligned_frame_ids: List[int] = field(default_factory=list)
    graph_node_ids: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        def convert(obj):
            if obj is None:
                return None
            if hasattr(obj, "__dict__"):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        return convert(self)
