from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class CommandIntent(str, Enum):
    QUESTION = "question"
    DESCRIBE_SCENE = "describe_scene"
    DETECT_PERSON = "detect_person"
    SCENE_BRIGHTNESS = "scene_brightness"
    SCENE_CROWD = "scene_crowd"
    SCENE_LOCATION_TYPE = "scene_location_type"
    TAKE_SNAPSHOT = "take_snapshot"
    STORE_MEMORY = "store_memory"
    START_STREAM = "start_stream"
    STOP_STREAM = "stop_stream"
    WHAT_IS_HEARD = "what_is_heard"
    SPEAKER_VISIBLE = "speaker_visible"
    OTHER = "other"

@dataclass
class TextSourceSpan:
    source_type: str
    source_id: str
    t0_ms: Optional[int] = None
    t1_ms: Optional[int] = None
    token_confidence: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntentScore:
    intent: CommandIntent
    confidence: float
    arguments: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmbeddingResult:
    enabled: bool
    model_name: str
    dimension: int
    vector_path: Optional[str]
    l2_norm: Optional[float]
    confidence: float

@dataclass
class TextSegPolicy:
    min_asr_confidence: float
    min_command_confidence: float
    max_tokens: int

@dataclass
class TextSegConfig:
    policy: TextSegPolicy

@dataclass
class TextSeg:
    seg_id: str
    created_at_ms: int
    text_original: str
    text_normalized: str
    language: str
    source: TextSourceSpan
    intents: List[IntentScore] = field(default_factory=list)
    embedding: Optional[EmbeddingResult] = None
    accepted: bool = True
    rejection_reason: Optional[str] = None
    links: Dict[str, Any] = field(default_factory=dict)

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
            if isinstance(obj, Enum):
                return obj.value
            return obj
        return convert(self)
