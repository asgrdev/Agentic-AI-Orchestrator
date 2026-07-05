# reasoning/llm_reasoner.py
"""
LLMReasoner - لایه استدلال Graph RAG

وظایف:
    1. دریافت context از Graph + Vector
    2. اجرای Chain-of-Thought reasoning
    3. تولید پاسخ نهایی با citation
    4. Citation Grounding: هر ادعا → chunk منبع

Backend های پشتیبانی‌شده:
    - MlxGraniteClient  : اجرای local روی Apple Silicon (sync)
    - GraniteClient     : اجرای remote async (آینده)
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from llm.granite_client import BaseGraniteClient, LLMResponse


# ══════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════

@dataclass
class Citation:
    """یک منبع استناد شده در پاسخ"""
    citation_id: str          # [1], [2], ...
    chunk_id: str             # شناسه chunk در Weaviate
    content_preview: str      # 120 کاراکتر اول محتوا
    score: float              # امتیاز relevance
    source_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "content_preview": self.content_preview,
            "score": self.score,
            "source_metadata": self.source_metadata,
        }


@dataclass
class ReasoningStep:
    """یک قدم از زنجیره استدلال"""
    step_number: int
    thought: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class CitedAnswer:
    """پاسخ نهایی با citation grounding کامل"""
    query: str
    answer: str
    citations: list[Citation]
    reasoning_steps: list[ReasoningStep]
    confidence: float
    entity_context: list[dict]
    relation_context: list[dict]

    def format_with_citations(self) -> str:
        lines = [
            f"**Answer:** {self.answer}",
            f"\n**Confidence:** {self._confidence_label()}",
        ]
        if self.citations:
            lines.append("\n**Sources:**")
            for cit in self.citations:
                meta = cit.source_metadata
                source_info = (
                    meta.get("source")
                    or meta.get("title")
                    or cit.chunk_id
                )
                lines.append(
                    f"  {cit.citation_id} {source_info} "
                    f"(relevance: {cit.score:.2f})"
                )
                lines.append(f"     *{cit.content_preview}*")
        return "\n".join(lines)

    def _confidence_label(self) -> str:
        if self.confidence >= 0.85:
            return f"HIGH ({self.confidence:.0%})"
        if self.confidence >= 0.55:
            return f"MEDIUM ({self.confidence:.0%})"
        return f"LOW ({self.confidence:.0%})"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence,
            "citations": [c.to_dict() for c in self.citations],
            "reasoning_steps": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "evidence": s.evidence,
                }
                for s in self.reasoning_steps
            ],
            "entity_context_count": len(self.entity_context),
            "relation_context_count": len(self.relation_context),
        }


# ══════════════════════════════════════════════════════════════════
# Citation Grounder
# ══════════════════════════════════════════════════════════════════

class CitationGrounder:
    """
    تطبیق [N] در پاسخ با search_results.

    استراتژی:
        1. پیدا کردن [N] در متن پاسخ
        2. تطبیق با search_results[N-1]
        3. اگر LLM citation نداد → auto-cite از top-3
    """

    _CITE_RE = re.compile(r"\[(\d+)\]")

    def ground(
        self,
        answer_text: str,
        search_results: list[Any],
    ) -> tuple[str, list[Citation]]:
        """
        Returns:
            (answer_text, citations)
            ─ answer_text ممکن است citation اضافه شده داشته باشد
        """
        used: set[int] = set()
        citations: list[Citation] = []

        for num_str in self._CITE_RE.findall(answer_text):
            idx = int(num_str) - 1
            if idx < 0 or idx >= len(search_results) or idx in used:
                continue
            used.add(idx)
            citations.append(
                self._make_citation(f"[{num_str}]", search_results[idx])
            )

        # اگر LLM هیچ citation ای نداد
        if not citations and search_results:
            citations = [
                self._make_citation(f"[{i}]", r)
                for i, r in enumerate(search_results[:3], 1)
            ]
            refs = " ".join(c.citation_id for c in citations)
            answer_text = f"{answer_text} {refs}"

        return answer_text, citations

    # ── private ───────────────────────────────────────────────────

    def _make_citation(self, citation_id: str, result: Any) -> Citation:
        if hasattr(result, "content"):
            content  = result.content
            chunk_id = getattr(result, "chunk_id", "unknown")
            score    = getattr(result, "score", 0.0)
            metadata = getattr(result, "metadata", {})
        else:
            content  = result.get("content", result.get("text", ""))
            chunk_id = result.get("chunk_id", "unknown")
            score    = result.get("score", 0.0)
            metadata = result.get("metadata", {})

        preview = content[:120].replace("\n", " ")
        if len(content) > 120:
            preview += "..."

        return Citation(
            citation_id=citation_id,
            chunk_id=chunk_id,
            content_preview=preview,
            score=score,
            source_metadata=metadata,
        )


# ══════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════

_SYS_REASONING = (
    "You are a precise, factual AI assistant with access to a knowledge graph. "
    "Reason step by step using the provided context only. "
    "Cite sources as [N] matching document numbers. "
    "Do NOT invent facts not present in the context."
)

_SYS_ANSWER = (
    "You are a precise, citation-aware AI assistant. "
    "Answer based ONLY on the provided context. "
    "Use [N] citations. End with 'CONFIDENCE: HIGH / MEDIUM / LOW'."
)

# برای reason_step
_STEP_PROMPT = """\
Question: {query}

Previous reasoning steps:
{prev_steps}

Context:
{context}

Write the next reasoning step. Be specific and cite evidence with [N] if applicable.
Focus on: what do we know, what is still unclear, what can we infer?
"""

# برای generate_answer
_ANSWER_PROMPT = """\
## Knowledge Graph - Entities
{entity_context}

## Knowledge Graph - Relations
{relation_context}

## Retrieved Documents
{doc_context}

## Reasoning Chain
{reasoning_trace}

## Question
{query}

## Instructions
- Use ONLY the context above
- Cite sources as [1], [2], etc.
- Be concise and factual
- End your answer with exactly one of:
  CONFIDENCE: HIGH
  CONFIDENCE: MEDIUM
  CONFIDENCE: LOW

## Answer
"""


# ══════════════════════════════════════════════════════════════════
# LLM Reasoner
# ══════════════════════════════════════════════════════════════════

class LLMReasoner:
    """
    لایه استدلال اصلی Graph RAG.

    استفاده:
        reasoner = LLMReasoner(client=mlx_client, context_builder=cb)
        answer   = await reasoner.reason(query, nodes, relations, results)
    """

    def __init__(
        self,
        client: BaseGraniteClient,
        context_builder: Any,
        *,
        max_reasoning_steps: int = 3,
        use_chain_of_thought: bool = True,
    ) -> None:
        """
        Args:
            client:               MlxGraniteClient یا هر BaseGraniteClient دیگر
            context_builder:      instance از ContextBuilder
            max_reasoning_steps:  حداکثر قدم‌های CoT
            use_chain_of_thought: فعال/غیرفعال کردن CoT
        """
        self.client   = client
        self.ctx      = context_builder
        self.max_steps = max_reasoning_steps
        self.use_cot   = use_chain_of_thought
        self.grounder  = CitationGrounder()

        # آیا client دارای متد async answer() است؟ (GraniteClient آینده)
        self._has_async_answer = (
            hasattr(client, "answer")
            and asyncio.iscoroutinefunction(getattr(client, "answer"))
        )
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
    # ── Public entry point ────────────────────────────────────────

    async def reason(
        self,
        query: str,
        graph_nodes: list[dict],
        graph_relations: list[dict],
        search_results: list[Any],
    ) -> CitedAnswer:
        """
        اجرای کامل pipeline استدلال.

        Args:
            query:           سوال کاربر
            graph_nodes:     nodes از KuzuDB.get_subgraph()
            graph_relations: relations از KuzuDB.get_subgraph()
            search_results:  نتایج از WeaviateStore.graph_rag_search()

        Returns:
            CitedAnswer کامل
        """
        # ۱. ساخت context strings
        entity_ctx   = self.ctx.format_graph_context(graph_nodes, graph_relations)
        doc_ctx      = self.ctx.format_doc_context(search_results)

        # ۲. Chain-of-Thought (در thread pool چون client.reason_step sync است)
        reasoning_steps: list[ReasoningStep] = []
        if self.use_cot:
            reasoning_steps = await asyncio.get_running_loop().run_in_executor(
                None,
                self._run_cot_sync,
                query,
                entity_ctx,
                doc_ctx,
            )

        # ۳. پاسخ نهایی
        reasoning_trace = self._format_trace(reasoning_steps)
        raw_answer, confidence = await asyncio.get_running_loop().run_in_executor(
            None,
            self._generate_answer_sync,
            query,
            entity_ctx,
            doc_ctx,
            reasoning_trace,
        )

        # ۴. Citation grounding
        grounded_answer, citations = self.grounder.ground(
            raw_answer, search_results
        )

        return CitedAnswer(
            query=query,
            answer=grounded_answer,
            citations=citations,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            entity_context=graph_nodes,
            relation_context=graph_relations,
        )

    # ── Chain-of-Thought (sync, در thread pool اجرا می‌شود) ──────

    def _run_cot_sync(
        self,
        query: str,
        entity_ctx: str,
        doc_ctx: str,
    ) -> list[ReasoningStep]:
        """
        اجرای sync Chain-of-Thought.
        در thread pool صدا زده می‌شود تا event loop بلاک نشود.
        """
        combined_context = f"{entity_ctx}\n\n{doc_ctx}"
        steps: list[ReasoningStep] = []
        prev_steps: list[str] = []

        for step_num in range(1, self.max_steps + 1):
            try:
                # ساخت prompt برای این قدم
                steps_text = (
                    "\n".join(
                        f"Step {i+1}: {s}" for i, s in enumerate(prev_steps)
                    )
                    or "None yet."
                )

                prompt = _STEP_PROMPT.format(
                    query=query,
                    prev_steps=steps_text,
                    # محدود کردن context برای جلوگیری از overflow
                    context=combined_context[:3000],
                )

                # صدا زدن client.reason_step که در BaseGraniteClient تعریف شده
                response: LLMResponse = self.client.reason_step(
                    question=query,
                    context=combined_context[:3000],
                    step_hint=f"reasoning step {step_num} of {self.max_steps}",
                )

                thought = response.text.strip()
                evidence = _extract_cite_refs(thought)

                steps.append(ReasoningStep(
                    step_number=step_num,
                    thought=thought,
                    evidence=evidence,
                ))
                prev_steps.append(thought)

                # اگر LLM نشان دهد به نتیجه رسیده، متوقف شو
                if _is_conclusion(thought):
                    break

            except Exception as exc:  # noqa: BLE001
                steps.append(ReasoningStep(
                    step_number=step_num,
                    thought=f"[Step failed: {exc}]",
                ))
                break

        return steps

    # ── Answer Generation (sync) ──────────────────────────────────

    def _generate_answer_sync(
        self,
        query: str,
        entity_ctx: str,
        doc_ctx: str,
        reasoning_trace: str,
    ) -> tuple[str, float]:
        """
        تولید پاسخ نهایی با استفاده از BaseGraniteClient.generate().
        """
        # entity_ctx شامل هر دو nodes و relations است (از format_graph_context)
        prompt = _ANSWER_PROMPT.format(
            entity_context=entity_ctx,
            # اگر ContextBuilder relations را جداگانه برمی‌گرداند، اینجا خالی
            relation_context="(see entity context above)",
            doc_context=doc_ctx,
            reasoning_trace=reasoning_trace,
            query=query,
        )

        response: LLMResponse = self.client.generate(
            prompt=prompt,
            system=_SYS_ANSWER,
        )

        text = response.text.strip()
        confidence = _extract_confidence(text)

        # حذف خط CONFIDENCE از انتهای پاسخ
        text = _CONFIDENCE_RE.sub("", text).strip()

        return text, confidence

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _format_trace(steps: list[ReasoningStep]) -> str:
        if not steps:
            return "No prior reasoning."
        return "\n".join(
            f"Step {s.step_number}: {s.thought}" for s in steps
        )


# ══════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════

_CITE_RE        = re.compile(r"\[(\d+)\]")
_CONFIDENCE_RE  = re.compile(
    r"\bCONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)\b",
    re.IGNORECASE,
)


def _extract_cite_refs(text: str) -> list[str]:
    return [f"[{m}]" for m in _CITE_RE.findall(text)]


def _is_conclusion(thought: str) -> bool:
    signals = {
        "therefore", "in conclusion", "the answer is",
        "based on the evidence", "to summarize",
        "بنابراین", "در نتیجه", "خلاصه اینکه",
    }
    lower = thought.lower()
    return any(s in lower for s in signals)


def _extract_confidence(text: str) -> float:
    m = _CONFIDENCE_RE.search(text)
    if not m:
        return 0.50
    level = m.group(1).upper()
    return {"HIGH": 0.90, "MEDIUM": 0.65, "LOW": 0.35}[level]
