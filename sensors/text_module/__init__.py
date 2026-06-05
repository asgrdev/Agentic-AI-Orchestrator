"""TextSeg module: local text understanding + embeddings for multimodal graph fusion."""
from .schema.textseg import TextSeg, TextSegPolicy, TextSegConfig, CommandIntent, TextSourceSpan
from .runtime.pipeline import TextSegPipeline
