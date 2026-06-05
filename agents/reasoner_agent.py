# reasoning/reasoner_agent.py
"""
ReasonerAgent - مرحله REASON در orchestrator

وظیفه:
    - دریافت context از AgentState.retrieval
    - صدا زدن LLMReasoner.reason()
    - ذخیره نتیجه در AgentState
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from agents.state import AgentState, FlowStep
from llm.reasoning.llm_reasoning import CitedAnswer, LLMReasoner
#from llm.reasoning.context_builder import ContextBuilder
from ingestion.context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class ReasonerAgent:

    def __init__(self, config: dict=[]) -> None:
        """
        config keys:
            llm_client      : BaseGraniteClient (باید از قبل startup شده باشد)
            context_builder : ContextBuilder (اختیاری، اگر نبود می‌سازیم)
            reasoner:
                max_reasoning_steps : int  (پیش‌فرض 3)
                use_chain_of_thought: bool (پیش‌فرض True)
        """
        reasoner_cfg = config.get("reasoner", {})

        ctx_builder: ContextBuilder = config.get(
            "context_builder", ContextBuilder()
        )
        if "llm_client_factory" in config:
            llm_client = config["llm_client_factory"]()
        else:
            llm_client = config["llm_client"]

  
        self._reasoner = LLMReasoner(
            client=llm_client,
            context_builder=ctx_builder,
            max_reasoning_steps=reasoner_cfg.get("max_reasoning_steps", 3),
            use_chain_of_thought=reasoner_cfg.get("use_chain_of_thought", True),
        )
        self._ready = True

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def startup(self) -> None:
        self._ready = True
        logger.info("ReasonerAgent ready.")

    async def shutdown(self) -> None:
        self._ready = False

    # ── Main Entry Point ───────────────────────────────────────────────

    async def reason(self, state: AgentState) -> AgentState:
        if not self._ready:
            raise RuntimeError("Call startup() first")

        retrieval = state.retrieval

        #این ممکن است بعد از تست حذف شود ، بدون وکتور ممکن است گراف باشد و ریزونینگ از آن استفاده کند
        #if not retrieval or not retrieval.vector_chunks:
           # logger.warning("No retrieval context — skipping reasoning.")
           # state.final_answer = "Could not retrieve sufficient context."
           # state.confidence = 1.0
           # return state

        try:
            cited: CitedAnswer = await self._reasoner.reason(
                query=state.query,
                graph_nodes=retrieval.graph_nodes,
                graph_relations=retrieval.graph_edges,
                search_results=retrieval.vector_chunks,
            )

            state.final_answer = cited.answer
            state.confidence = cited.confidence
            state.citations = [c.to_dict() for c in cited.citations]
            state.reasoning_trace = [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "evidence": s.evidence,
                }
                for s in cited.reasoning_steps
            ]
            logger.info(state.final_answer)
            
            logger.info(state.citations)
            
            logger.info(state.reasoning_trace)
            
            # fused_context را هم اینجا می‌سازیم (کار ReasonerAgent است)
            retrieval.fused_context = cited.format_with_citations()

            logger.info(
                "Reasoning done | confidence=%.2f citations=%d steps=%d",
                state.confidence,
                len(state.citations),
                len(state.reasoning_trace),
            )

        except Exception as e:
            logger.error("LLM reasoning failed: %s", e, exc_info=True)
            state.final_answer = f"Error during reasoning: {e}"
            state.confidence = 0.0
            state.current_step = FlowStep.ERROR

        return state
