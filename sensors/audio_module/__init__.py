"""AudioSeg module: EfficientAT + Whisper pipeline with local/offline models."""

from .schema.audioseg import (
    AudioSeg,
    AudioSegConfig,
    AudioSegPolicy,
    VadResult,
    TaggingResult,
    EmbeddingResult,
    AsrResult,
)
from .runtime.pipeline import AudioSegPipeline
