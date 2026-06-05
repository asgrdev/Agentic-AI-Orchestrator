from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import uuid
import logging
import numpy as np

from ..config import RuntimeConfig
from ..schema.audioseg import AudioSeg, AudioQuality, VadResult, TagItem, TaggingResult, EmbeddingResult, AsrResult, AsrWord
from ..sources.base import AudioChunk
from ..segment.segmenter import StreamSegmenter, RawAudioSegment
from ..processors.quality import AudioQualityEstimator
from ..processors.vad import EnergyVad
from ..processors.efficientat import PANNsProcessor, PANNsConfig
from ..processors.whisper_asr import WhisperProcessor, WhisperConfig
from ..processors.policy import PolicyEngine
from ..sinks.embedding_store import EmbeddingStore

logger = logging.getLogger("audio_module")

class AudioSegPipeline:
    def __init__(self, runtime: RuntimeConfig, efficientat_cfg: PANNsConfig, whisper_cfg: WhisperConfig, embedding_store_dir: str) -> None:
        self.runtime = runtime
        self.dataset_resources = dict(getattr(runtime, "dataset_resources", {}) or {})
        self.segmenter = StreamSegmenter(
            sample_rate_hz=runtime.sample_rate_hz,
            segment_duration_sec=runtime.segment_duration_sec,
            hop_sec=runtime.segment_hop_sec,
            max_buffer_seconds=runtime.max_buffer_seconds,
        )
        self.quality_estimator = AudioQualityEstimator()
        self.vad = EnergyVad(enabled=True, speech_probability_threshold=runtime.vad_speech_prob_min)
        self.efficientat = PANNsProcessor(efficientat_cfg)
        self.whisper = WhisperProcessor(whisper_cfg)
        self.policy = PolicyEngine(
            vad_speech_prob_min=runtime.vad_speech_prob_min,
            tag_speech_prob_min=runtime.tag_speech_prob_min,
            max_noise_tag_prob=runtime.max_noise_tag_prob,
            asr_min_gap_sec=runtime.asr_min_gap_sec,
        )
        self.embedding_store = EmbeddingStore(embedding_store_dir)
        self._last_tag_ms: Optional[int] = None
        self._last_embed_ms: Optional[int] = None

    def process_chunk(self, chunk: AudioChunk, user_requested_asr: bool = False) -> List[AudioSeg]:
        segments: List[AudioSeg] = []
        for raw_seg in self.segmenter.push_chunk(chunk):
            segments.append(self._process_segment(raw_seg, user_requested_asr))
        return segments

    def _should_run_periodic(self, last_ms: Optional[int], every_sec: float, now_ms: int) -> bool:
        if every_sec <= 0:
            return False
        if last_ms is None:
            return True
        return (now_ms - last_ms) >= int(every_sec * 1000)

    def _process_segment(self, raw: RawAudioSegment, user_requested_asr: bool) -> AudioSeg:
        seg_id = str(uuid.uuid4())
        now_ms = raw.t1_ms

        q = self.quality_estimator.estimate(raw.pcm)
        quality = AudioQuality(**q)

        vad_prob, speech_flag = self.vad.infer(raw.pcm)
        vad = VadResult(enabled=self.vad.enabled, speech_probability=float(vad_prob), speech_detected=bool(speech_flag), threshold=float(self.vad.speech_probability_threshold))

        tags_result: Optional[TaggingResult] = None
        embedding_result: Optional[EmbeddingResult] = None

        tag_topk_pairs: Optional[List[tuple[str, float]]] = None
        run_tag = self._should_run_periodic(self._last_tag_ms, self.runtime.tagging_every_sec, now_ms)
        run_embed = self._should_run_periodic(self._last_embed_ms, self.runtime.embedding_every_sec, now_ms)

        if run_tag or run_embed:
            topk, entropy, conf, embedding = self.efficientat.infer(raw.pcm, now_ms)
            tag_topk_pairs = topk

            if run_tag:
                self._last_tag_ms = now_ms
                tags_result = TaggingResult(
                    model_name=self.efficientat.cfg.model_name,
                    enabled=True,
                    topk=[TagItem(label=l, probability=p) for l, p in topk[: self.runtime.topk_tags]],
                    tag_entropy=float(entropy),
                    confidence=float(conf),
                )

            if run_embed:
                self._last_embed_ms = now_ms
                if embedding is not None:
                    vector_path = self.embedding_store.save(seg_id, embedding)
                    l2 = float(np.linalg.norm(embedding) + 1e-12)
                    embedding_result = EmbeddingResult(
                        model_name=self.efficientat.cfg.model_name,
                        enabled=True,
                        dimension=int(embedding.shape[0]),
                        vector_path=vector_path,
                        l2_norm=l2,
                    )
                else:
                    embedding_result = EmbeddingResult(model_name=self.efficientat.cfg.model_name, enabled=True, dimension=0)

        decision = self.policy.decide_asr(
            now_ms=now_ms,
            user_requested=user_requested_asr,
            vad_prob=float(vad_prob),
            tag_topk=tag_topk_pairs,
        )

        asr_obj = AsrResult(
            model_name=f"whisper-{self.whisper.cfg.model_name}",
            enabled=True,
            ran=False,
            run_reason=decision.reason,
            language=self.whisper.cfg.language,
            temperature=self.whisper.cfg.temperature,
        )

        if decision.should_run and self.whisper.can_run(now_ms, self.runtime.asr_min_gap_sec):
            try:
                result = self.whisper.transcribe(raw.pcm, raw.sample_rate_hz, now_ms)
                asr_obj.ran = True
                asr_obj.text = (result.get("text", "") or "").strip()
                asr_obj.avg_logprob = result.get("avg_logprob", None)
                asr_obj.no_speech_prob = result.get("no_speech_prob", None)
                asr_obj.compression_ratio = result.get("compression_ratio", None)
                asr_obj.temperature = result.get("temperature", self.whisper.cfg.temperature)

                words: List[AsrWord] = []
                for seg in result.get("segments", []) or []:
                    for w in seg.get("words", []) or []:
                        words.append(AsrWord(
                            word=str(w.get("word", "")).strip(),
                            start_sec=float(w.get("start", 0.0)),
                            end_sec=float(w.get("end", 0.0)),
                            probability=w.get("probability", None),
                        ))
                asr_obj.words = words
            except Exception as exc:
                logger.exception("Whisper transcription failed: %s", exc)

        return AudioSeg(
            seg_id=seg_id,
            source_id=raw.source_id,
            t0_ms=int(raw.t0_ms),
            t1_ms=int(raw.t1_ms),
            sample_rate_hz=int(raw.sample_rate_hz),
            n_samples=int(raw.pcm.shape[0]),
            quality=quality,
            vad=vad,
            tags=tags_result,
            embedding=embedding_result,
            asr=asr_obj,
        )
