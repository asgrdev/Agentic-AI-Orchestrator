from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AsrDecision:
    should_run: bool
    reason: str

class PolicyEngine:
    def __init__(
        self,
        vad_speech_prob_min: float,
        tag_speech_prob_min: float,
        max_noise_tag_prob: float,
        asr_min_gap_sec: float,
    ) -> None:
        self.vad_speech_prob_min = vad_speech_prob_min
        self.tag_speech_prob_min = tag_speech_prob_min
        self.max_noise_tag_prob = max_noise_tag_prob
        self.asr_min_gap_sec = asr_min_gap_sec
        self._last_asr_ms: Optional[int] = None

    def _within_gap(self, now_ms: int) -> bool:
        if self._last_asr_ms is None:
            return False
        return (now_ms - self._last_asr_ms) < int(self.asr_min_gap_sec * 1000)

    def decide_asr(self, now_ms: int, user_requested: bool, vad_prob: float, tag_topk: Optional[List[tuple[str, float]]]) -> AsrDecision:
        if self._within_gap(now_ms) and not user_requested:
            return AsrDecision(False, "min_gap")
        if user_requested:
            self._last_asr_ms = now_ms
            return AsrDecision(True, "user_request")
        if vad_prob < self.vad_speech_prob_min:
            return AsrDecision(False, "vad_gate")

        if tag_topk:
            speech_prob = 0.0
            max_noise = 0.0
            for label, prob in tag_topk:
                name = label.lower()
                if "speech" in name or "talk" in name or "conversation" in name:
                    speech_prob = max(speech_prob, prob)
                if any(k in name for k in ["music", "engine", "wind", "alarm", "sir", "vehicle", "traffic"]):
                    max_noise = max(max_noise, prob)

            if max_noise >= self.max_noise_tag_prob and speech_prob < self.tag_speech_prob_min:
                return AsrDecision(False, "noise_veto")
            if speech_prob >= self.tag_speech_prob_min:
                self._last_asr_ms = now_ms
                return AsrDecision(True, "speech_confirmed")

        self._last_asr_ms = now_ms
        return AsrDecision(True, "vad_only")
