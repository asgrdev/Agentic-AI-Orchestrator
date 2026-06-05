# agents/validator_agent.py
"""
ValidatorAgent - مرحله VALIDATE در orchestrator

وظیفه:
    - بررسی کیفیت پاسخ تولیدشده توسط ReasonerAgent
    - سنجش انسجام پاسخ با context بازیابی‌شده
    - تولید score برای تصمیم‌گیری orchestrator:
        score >= 0.75  →  ANSWER
        score >= 0.50  →  RETRIEVE (بازیابی مجدد)
        score <  0.50  →  REFRESH  (رفرش دانش)
"""
from __future__ import annotations

import json
import logging
import re
import numpy as np
from agents.state import AgentState, FlowStep

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

VALIDATE_SYSTEM = """\
You are a strict answer quality validator for an AgenticGraphRAG system.

Your job:
1. Check if the answer directly addresses the query
2. Verify the answer is grounded in the provided context (no hallucination)
3. Check citation coverage — key claims must be supported
4. Assess completeness — are all aspects of the query addressed?
5. Detect contradictions between the answer and context

Scoring rubric:
- 0.90-1.00 : Perfect — fully grounded, complete, no contradictions
- 0.75-0.89 : Good    — mostly grounded, minor gaps acceptable
- 0.50-0.74 : Partial — answer exists but missing key aspects or weak grounding
- 0.00-0.49 : Poor    — hallucinated, off-topic, or contradicts context

Rules:
- Respond ONLY with valid JSON, no markdown, no explanation
- Be strict: penalize any claim not traceable to context
"""

VALIDATE_TEMPLATE = """\
Query: {query}
Intent: {intent}

Retrieved context (truncated):
{context}

Generated answer:
{answer}

Citations provided: {citation_count}

Respond with this exact JSON schema:
{{
  "score": 0.85,
  "is_grounded": true,
  "is_complete": true,
  "has_contradictions": false,
  "missing_aspects": ["..."],
  "hallucinated_claims": ["..."],
  "feedback": "brief explanation",
  "refined_answer": "improved answer if score < 0.75, else same as input"
}}
"""


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON in validator output:\n{text[:300]}")


def _build_context_snippet(state: AgentState, max_chars: int = 2000) -> str:
    """
    بهترین context موجود را برای validation انتخاب می‌کند.
    اولویت: fused_context > vector_chunks > bm25_docs
    """
    r = state.retrieval

    if r.fused_context:
        return r.fused_context[:max_chars]

    if r.vector_chunks:
        return "\n".join(
            ch.get("content", "") for ch in r.vector_chunks[:5]
        )[:max_chars]

    if r.bm25_docs:
        return "\n".join(
            d.get("content", "") for d in r.bm25_docs[:5]
        )[:max_chars]

    return ""


# ─────────────────────────────────────────────
# ValidatorAgent
# ─────────────────────────────────────────────

class ValidatorAgent:
    """
    پاسخ تولیدشده توسط ReasonerAgent را ارزیابی می‌کند.

    config keys:
        phi4_mini_client : Phi4MiniClient (اشتراکی با Orchestrator)
        validator:
            score_threshold_answer  : float  (پیش‌فرض 0.75)
            score_threshold_retrieve: float  (پیش‌فرض 0.50)
            min_answer_length       : int    (پیش‌فرض 10)
    """

    def __init__(self, config: dict) -> None:
        # phi4-mini برای validation استفاده می‌شود — همان client اشتراکی
        self._phi4 = config.get("phi4_mini_client")

        cfg = config.get("validator", {})
        self._threshold_answer   = cfg.get("score_threshold_answer",   0.75)
        self._threshold_retrieve = cfg.get("score_threshold_retrieve", 0.50)
        self._min_answer_len     = cfg.get("min_answer_length",        10)
        self._ready = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def startup(self) -> None:
        self._ready = True
        logger.info("ValidatorAgent ready.")

    async def shutdown(self) -> None:
        self._ready = False

    # ── Main Entry Point ───────────────────────────────────────────────

    async def validate(self, state: AgentState) -> dict:
        """
        خروجی:
        {
            "score"  : float,   # امتیاز کلی [0, 1]
            "answer" : str,     # پاسخ نهایی (refined اگر لازم بود)
            "feedback": str,
            "missing_aspects": list[str],
            "hallucinated_claims": list[str],
        }
        """
        answer = state.final_answer

        # ── بررسی‌های اولیه بدون LLM ──────────────────────────────────
        quick_fail = self._quick_checks(answer, state)
        if quick_fail is not None:
            return quick_fail

        # ── LLM-based validation ───────────────────────────────────────
        if self._phi4 is not None:
            return await self._llm_validate(state, answer)

        # ── fallback: heuristic scoring ───────────────────────────────
        logger.warning("No phi4_mini_client — using heuristic validation.")
        return self._heuristic_validate(state, answer)

    # ── Internal Methods ───────────────────────────────────────────────

    def _quick_checks(self, answer: str, state: AgentState) -> dict | None:
        """
        بررسی‌های سریع که نیازی به LLM ندارند.
        اگر مشکل جدی پیدا شد، مستقیم نتیجه برمی‌گرداند.
        """
        if not answer or len(answer.strip()) < self._min_answer_len:
            logger.warning("Answer too short or empty.")
            return {
                "score": 0.0,
                "answer": answer,
                "feedback": "Answer is empty or too short.",
                "missing_aspects": ["complete answer"],
                "hallucinated_claims": [],
            }

        if state.current_step == FlowStep.ERROR:
            return {
                "score": 0.0,
                "answer": answer,
                "feedback": "Pipeline is in ERROR state.",
                "missing_aspects": [],
                "hallucinated_claims": [],
            }

        return None  # ادامه به validation اصلی

    async def _llm_validate(self, state: AgentState, answer: str) -> dict:
        context = _build_context_snippet(state)

        prompt = VALIDATE_TEMPLATE.format(
            query=state.query,
            intent=state.intent or "factual",
            context=context or "(no context available)",
            answer=answer,
            citation_count=len(state.citations),
        )

        try:
            raw = await self._phi4._backend.chat(
                system=VALIDATE_SYSTEM,
                user=prompt,
                temperature=0.0,
                max_tokens=512,
            )
            data = _extract_json(raw)
            return self._normalize_result(data, answer)

        except Exception as e:
            logger.warning(f"LLM validation failed: {e} — falling back to heuristic.")
            return self._heuristic_validate(state, answer)

    def _heuristic_validate(self, state: AgentState, answer: str) -> dict:
        """
        وقتی LLM در دسترس نیست:
        - confidence از ReasonerAgent را مستقیم استفاده می‌کنیم
        - یک بررسی ساده keyword overlap انجام می‌دهیم
        """
        score = state.confidence  # از reasoner

        # keyword overlap بین query و answer
        query_words = set(state.query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words & answer_words) / max(len(query_words), 1)

        # اگر overlap خیلی کم بود، score را کاهش بده
        if overlap < 0.1:
            score = min(score, 0.45)

        # اگر citation داشت، کمی امتیاز بده
        if state.citations:
            score = min(1.0, score + 0.05)

        return {
            "score": round(score, 3),
            "answer": answer,
            "feedback": f"Heuristic validation | overlap={overlap:.2f} | citations={len(state.citations)}",
            "missing_aspects": state.knowledge_gaps or [],
            "hallucinated_claims": [],
        }

    @staticmethod
    def _normalize_result(data: dict, original_answer: str) -> dict:
        """
        خروجی LLM را normalize و sanitize می‌کند.
        """
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # clamp به [0, 1]

        # اگر score پایین بود و refined_answer وجود داشت، از آن استفاده کن
        refined = data.get("refined_answer", "").strip()
        answer = (
            refined
            if score < 0.75 and refined and len(refined) > 10
            else original_answer
        )

        return {
            "score": round(score, 3),
            "answer": answer,
            "feedback": data.get("feedback", ""),
            "missing_aspects": data.get("missing_aspects", []),
            "hallucinated_claims": data.get("hallucinated_claims", []),
        }


    def semantic_score(answer: str, context: str, embedder) -> float:
        emb_answer  = embedder.encode(answer)
        emb_context = embedder.encode(context)
        
        # cosine similarity
        score = np.dot(emb_answer, emb_context) / (
            np.linalg.norm(emb_answer) * np.linalg.norm(emb_context)
        )
    
        return float(score)