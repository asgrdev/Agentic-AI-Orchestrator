from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import time, uuid
import numpy as np

from ..config import RuntimeConfig
from ..schema.textseg import TextSeg, TextSegPolicy, TextSourceSpan, EmbeddingResult
from ..sources.base import TextPacket
from ..processors.normalization import normalize_text
from ..processors.language import guess_language
from ..processors.command_parser import CommandParser
from ..processors.bert_embedder import BertEmbedder, BertEmbedderConfig
from ..sinks.embedding_store import EmbeddingStore

@dataclass
class ModelConfig:
    bert_name_or_path: str = "bert-base-cased"

class TextSegPipeline:
    def __init__(
        self,
        runtime: RuntimeConfig,
        models: Optional[ModelConfig] = None,
        device: str = "cpu",
        embedding_store_dir: str = "outputs/text_embeddings",
        bert_cfg: Optional[Any] = None,
    ) -> None:
        self.runtime = runtime
        self.command_parser = CommandParser()
        self.embedding_store = EmbeddingStore(embedding_store_dir)
        self.dataset_resources = dict(getattr(runtime, "dataset_resources", {}) or {})
        self._packet_id = 0

        model_name = (
            models.bert_name_or_path
            if models is not None
            else getattr(bert_cfg, "model_name_or_path", None)
            or getattr(bert_cfg, "model_path", None)
            or "bert-base-cased"
        )
        model_device = str(getattr(bert_cfg, "device", device) or device)
        max_tokens = int(getattr(bert_cfg, "max_tokens", runtime.max_tokens))

        if runtime.embedding_enabled:
            cfg = BertEmbedderConfig(
                model_name_or_path=str(model_name),
                device=model_device,
                max_tokens=max_tokens,
            )
            self.embedder = BertEmbedder(cfg)
        else:
            self.embedder = None

    def process_packet(self, pkt: TextPacket) -> TextSeg:
        seg_id = str(uuid.uuid4())
        text_original = pkt.text or ""
        text_normalized = normalize_text(text_original, normalize_whitespace=self.runtime.normalize_whitespace, strip_control_chars=self.runtime.strip_control_chars)
        source_span = TextSourceSpan(source_type=pkt.source_type, source_id=pkt.source_id, t0_ms=pkt.t0_ms, t1_ms=pkt.t1_ms, token_confidence=pkt.confidence, meta=pkt.meta or {})
        seg = TextSeg(seg_id=seg_id, created_at_ms=int(time.time()*1000), text_original=text_original, text_normalized=text_normalized,
                      language=guess_language(text_normalized) if self.runtime.language_detection_enabled else "unknown", source=source_span)

        if len(text_normalized) < self.runtime.min_text_length:
            seg.accepted = False
            seg.rejection_reason = "text_too_short"
            return seg

        if pkt.source_type == "asr":
            conf = float(pkt.confidence) if pkt.confidence is not None else 0.0
            if conf < self.runtime.min_asr_confidence:
                seg.accepted = False
                seg.rejection_reason = f"asr_conf_below_threshold:{conf:.2f}"
                return seg

        intents = self.command_parser.parse(text_normalized)
        seg.intents = [i for i in intents if i.confidence >= self.runtime.min_command_confidence]

        if self.embedder is not None:
            vec = self.embedder.embed(text_normalized)
            vec_path = self.embedding_store.save(seg_id, vec)
            base_conf = float(pkt.confidence) if pkt.confidence is not None else 0.5
            seg.embedding = EmbeddingResult(enabled=True, model_name=self.embedder.cfg.model_name, dimension=int(vec.shape[0]),
                                            vector_path=vec_path, l2_norm=float(np.linalg.norm(vec)+1e-12),
                                            confidence=float(np.clip(0.6 + 0.4*base_conf, 0.0, 1.0)))

        seg.links["policy"] = TextSegPolicy(min_asr_confidence=float(self.runtime.min_asr_confidence),
                                            min_command_confidence=float(self.runtime.min_command_confidence),
                                            max_tokens=int(self.runtime.max_tokens)).__dict__
        return seg

    def process_text(
        self,
        text: str,
        source_id: str = "text",
        source_type: str = "user_prompt",
        confidence: Optional[float] = None,
        t0_ms: Optional[int] = None,
        t1_ms: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> TextSeg:
        # Backward-compatible helper used by older callers.
        self._packet_id += 1
        pkt = TextPacket(
            packet_id=self._packet_id,
            created_at_ms=int(time.time() * 1000),
            text=text,
            source_type=source_type,
            source_id=source_id,
            t0_ms=t0_ms,
            t1_ms=t1_ms,
            confidence=confidence,
            meta=meta or {},
        )
        return self.process_packet(pkt)

    def close(self) -> None:
        return
