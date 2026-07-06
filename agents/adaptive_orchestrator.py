"""
Adaptive Orchestrator — هماهنگ‌کننده‌ی فلو، به سبک StateGraph در LangGraph.

گراف فلو (همان چیزی که tab «Flow» نمایش می‌دهد):

                          ┌─────────┐
                     ┌───▶│ REFRESH │────┐
                     │    └─────────┘    ▼
    START ─▶ UNDERSTAND ─▶ RETRIEVE ─▶ ASSESS ─▶ REASON ─▶ VALIDATE ─▶ ANSWER
      │                        ▲                    │          │  ▲       ▲
      │                        └────────────────────┼──────────┘  │       │
      └─▶ SHORTCUT (پاسخ مستقیم گفت‌وگو/محاسبه) ────┼─────────────┘───────┘
                                                    ▼
                                                  ERROR

معادل مفاهیم LangGraph در این کد:
┌──────────────────────┬──────────────────────────────────────────────────┐
│ مفهوم LangGraph      │ این‌جا                                           │
├──────────────────────┼──────────────────────────────────────────────────┤
│ State                │ agents/state.py → AgentState                     │
│ Node                 │ متدهای _step_* (جدول _STEP_HANDLERS پایین‌تر)    │
│ Conditional Edge     │ مقدار بازگشتی هر _step_* (FlowStep بعدی)         │
│ Graph definition     │ self._transitions (یال‌های مجاز)                 │
│ compile().invoke()   │ run() → _execute_dynamic_plan()                  │
│ Checkpointer         │ core/session_store.py (ذخیره‌ی trace هر اجرا)   │
│ recursion_limit      │ DynamicPlan.max_iterations                       │
└──────────────────────┴──────────────────────────────────────────────────┘

هر گره (node) یک متد async است که state را می‌گیرد، کارش را می‌کند و نام
گره‌ی بعدی را برمی‌گرداند — برای اضافه‌کردن گره‌ی جدید: متد _step_<name>
بسازید، به _STEP_HANDLERS و _transitions اضافه کنید و به FlowStep در
state.py یک مقدار بدهید. مسیر طی‌شده در state.flow_trace ثبت و در UI
نمایش داده می‌شود؛ «فکر» مدل‌ها هم در state.thinking_log جمع می‌شود.
"""
from __future__ import annotations

import re
import time
import logging
import gc
from dataclasses import dataclass, field
from typing import Optional

from agents.state import AgentState, FlowStep
from agents.query_shortcuts import is_conversational, extract_calc_expression
from agents.query_classifier import QueryClassifier, QueryAnalysis, QueryType, QueryComplexity
from agents.retriever_agent import RetrieverAgent
from agents.reasoner_agent import ReasonerAgent
from agents.validator_agent import ValidatorAgent
from agents.skill_executor import SkillExecutor, SkillResult, get_global_registry
from agents.improved_knowledge_refresh_agent import ImprovedKnowledgeRefreshAgent

logger = logging.getLogger(__name__)


@dataclass
class DynamicPlan:
    """Plan داینامیک برای اجرای query"""
    # مراحل
    required_steps: list[FlowStep] = field(default_factory=list)
    skip_steps: list[FlowStep] = field(default_factory=list)
    
    # کنترل تکرار
    max_iterations: int = 2
    max_retrieve_iterations: int = 2
    
    # فلگ‌ها
    force_refresh: bool = False
    skip_assessment: bool = False
    allow_early_exit: bool = False
    enable_sub_questions: bool = False
    
    # تنظیمات
    confidence_threshold: float = 0.75
    disable_graph_traversal: bool = False
    use_simple_answer: bool = False


class AdaptiveOrchestrator:
    """
    Orchestrator هوشمند با قابلیت کنترل داینامیک فلو
    
    ترکیب فلوی ثابت (reliability) با routing داینامیک (flexibility)
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Query Classifier
        self.query_classifier = QueryClassifier()
        
        # Skill Executor
        self.skill_executor = SkillExecutor(get_global_registry())
        
        # Agents
        self.retriever = RetrieverAgent(config)
        self.validator = ValidatorAgent(config)

        # refresher از همان connection های retriever استفاده می‌کند —
        # دو AsyncKuzuManager/WeaviateStore جدا یعنی دوبرابر حافظه و thread pool
        refresher_config = {
            **config,
            "shared_kuzu": self.retriever._kuzu,
            "shared_weaviate": self.retriever._weaviate,
        }
        self.refresher = ImprovedKnowledgeRefreshAgent(refresher_config)

        # جایگزینی skill های stub با پیاده‌سازی واقعی (vector/graph/web)
        self._register_real_skills()

        # آمار اجرای فلو برای dashboard/Flow — تاریخچه‌ی ذخیره‌شده‌ی
        # سشن‌های قبلی هم لود می‌شود تا بعد از restart از بین نرود
        from core.session_store import get_session_store
        self._session_store = get_session_store()
        self.flow_history: list[dict] = self._session_store.load_recent(200)
        # state کوئری در حال اجرا (یا آخرین) — برای نمایش زنده در tab «Flow»
        self.active_state: Optional[AgentState] = None
        
        # نگاشت انتقال‌های مجاز (برای validation)
        self._transitions: dict[FlowStep, list[FlowStep]] = {
            FlowStep.START:      [FlowStep.UNDERSTAND],
            FlowStep.UNDERSTAND: [FlowStep.RETRIEVE, FlowStep.REFRESH, FlowStep.REASON, FlowStep.ANSWER, FlowStep.ERROR],
            FlowStep.RETRIEVE:   [FlowStep.ASSESS, FlowStep.REASON],  # می‌تواند مستقیم به REASON برود
            FlowStep.ASSESS:     [FlowStep.REASON, FlowStep.REFRESH],
            FlowStep.REFRESH:    [FlowStep.RETRIEVE],
            FlowStep.REASON:     [FlowStep.VALIDATE, FlowStep.ANSWER],  # می‌تواند مستقیم به ANSWER برود
            FlowStep.VALIDATE:   [FlowStep.ANSWER, FlowStep.RETRIEVE, FlowStep.REFRESH],
            FlowStep.ANSWER:     [],
            FlowStep.ERROR:      [],
        }
    
    # ══════════════════════════════════════════════════════════════
    # Real Skill Wiring
    # ══════════════════════════════════════════════════════════════

    def _register_real_skills(self) -> None:
        """
        skill های پیش‌فرض registry همه stub هستند و [] برمی‌گردانند —
        اینجا با backend های واقعی (Weaviate/KuzuDB/DataCollector) جایگزین می‌شوند.
        """
        registry = self.skill_executor.registry
        orch = self

        async def real_search(query: str, sources: list = None, top_k: int = 5) -> dict:
            """جستجوی واقعی: hybrid vector + graph"""
            sources = sources or ["vector", "graph"]
            results: list[dict] = []
            try:
                if not orch.retriever._ready:
                    await orch.retriever.startup()

                if "vector" in sources:
                    embedding = await orch.retriever._embed(query)
                    hits = await orch.retriever._weaviate.hybrid_search(
                        query_text=query,
                        query_embedding=embedding,
                        top_k=top_k,
                    )
                    for h in hits:
                        results.append({
                            "source": "vector",
                            "content": getattr(h, "content", ""),
                            "score": getattr(h, "score", 0.0),
                            "chunk_id": getattr(h, "chunk_id", ""),
                        })

                if "graph" in sources:
                    res = await orch.retriever._kuzu.find_entity_by_name(query, limit=top_k)
                    for row in (res.rows if res else []):
                        results.append({"source": "graph", **dict(row)})
            except Exception as e:
                logger.warning("real_search failed: %s", e)
                return {"query": query, "sources": sources, "results": results, "error": str(e)}
            return {"query": query, "sources": sources, "results": results}

        async def real_retrieve(query: str, top_k: int = 5) -> dict:
            """بازیابی واقعی از vector store"""
            out = await real_search(query, sources=["vector"], top_k=top_k)
            return {"query": query, "top_k": top_k, "chunks": out["results"]}

        async def real_graph_query(cypher: str) -> dict:
            """اجرای Cypher واقعی روی KuzuDB"""
            try:
                if not orch.retriever._ready:
                    await orch.retriever.startup()
                res = await orch.retriever._kuzu.execute(cypher)
                return {"cypher": cypher, "results": list(res.rows) if res else []}
            except Exception as e:
                logger.warning("graph_query failed: %s", e)
                return {"cypher": cypher, "results": [], "error": str(e)}

        async def real_web_search(query: str, num_results: int = 5) -> dict:
            """جستجوی منابع خارجی (Wikipedia/arXiv/RSS) از طریق DataCollectorManager"""
            import asyncio as _asyncio
            try:
                items = await _asyncio.wait_for(
                    _asyncio.to_thread(
                        orch.refresher.data_manager.collect_knowledge,
                        query, num_results,
                    ),
                    timeout=30.0,
                )
                results = [
                    {
                        "title": it.title,
                        "content": (it.content or "")[:500],
                        "url": it.url,
                        "source": it.source,
                    }
                    for it in items[:num_results]
                ]
                return {"query": query, "num_results": num_results, "results": results}
            except Exception as e:
                logger.warning("web_search failed: %s", e)
                return {"query": query, "num_results": num_results, "results": [], "error": str(e)}

        registry.register(
            "search", real_search,
            "Search across vector store and knowledge graph (real backends)",
            {"query": "string", "sources": "list[string]", "top_k": "int"},
        )
        registry.register(
            "retrieve", real_retrieve,
            "Retrieve relevant chunks from the vector store (real backend)",
            {"query": "string", "top_k": "int"},
        )
        registry.register(
            "graph_query", real_graph_query,
            "Run a Cypher query against KuzuDB (real backend)",
            {"cypher": "string"},
        )
        registry.register(
            "web_search", real_web_search,
            "Search external sources (Wikipedia, arXiv, RSS) for fresh information",
            {"query": "string", "num_results": "int"},
        )
        logger.info("Real skills registered: search, retrieve, graph_query, web_search")

    # ══════════════════════════════════════════════════════════════
    # Main Entry Point
    # ══════════════════════════════════════════════════════════════

    async def run(self, query: str, session_id: str) -> AgentState:
        """
        اجرای adaptive با کنترل داینامیک
        """
        state = AgentState(query=query, session_id=session_id)
        state.current_step = FlowStep.START
        self._trace(state, "start", 0.0)
        # state فعال برای نمایش زنده در tab «Flow»
        self.active_state = state

        try:
            # مرحله 0: میان‌برها — «hello» یا «2+2» نباید کل RAG را طی کند
            # (قبلاً هر سلام ساده ~۲ دقیقه understand+retrieve+refresh می‌خورد)
            if await self._try_shortcut(state):
                self._record_flow(state)
                logger.info(
                    "Query answered via shortcut (%s) — full RAG flow skipped",
                    state.strategy,
                )
                return state

            # مرحله 1: UNDERSTAND + Query Analysis
            await self._step_understand_and_analyze(state)
            
            # مرحله 2: اجرای plan داینامیک
            if hasattr(state, 'dynamic_plan') and state.dynamic_plan:
                await self._execute_dynamic_plan(state)
            else:
                # fallback به فلوی ثابت
                await self._execute_fixed_flow(state)
            
        except Exception as e:
            logger.error(f"Orchestrator failed: {e}", exc_info=True)
            state.current_step = FlowStep.ERROR
            state.errors.append(str(e))
            self._trace(state, "error", 0.0, status="error", detail=str(e)[:120])
        finally:
            gc.collect()

        # حتی در حالت خطا هم پیام قابل نمایش برگردان
        if not state.final_answer:
            last_error = state.errors[-1] if state.errors else "unknown"
            state.final_answer = (
                "متأسفانه پردازش این سوال کامل نشد و پاسخی تولید نشد. "
                f"(مرحله: {state.current_step.value} | خطا: {last_error})"
            )

        if state.current_step == FlowStep.ANSWER and (
            not state.flow_trace or state.flow_trace[-1]["step"] != "answer"
        ):
            self._trace(state, "answer", 0.0, detail=f"confidence={state.confidence:.2f}")

        # ثبت در تاریخچه فلو (برای dashboard)
        self._record_flow(state)

        logger.info(f"Query completed: {state.current_step} | confidence: {state.confidence:.2f}")
        return state

    # ══════════════════════════════════════════════════════════════
    # Flow Trace — برای نمایش گراف اجرای زنده (api/flow_visualizer)
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _trace(
        state: AgentState,
        step: str,
        duration: float,
        status: str = "done",
        detail: str = "",
    ) -> None:
        state.flow_trace.append({
            "step": step,
            "duration": round(duration, 3),
            "status": status,
            "detail": detail,
            "ts": time.time(),
        })

    @staticmethod
    def _think(state: AgentState, stage: str, text: str) -> None:
        """ثبت یک قطعه‌ی «فکر» برای نمایش زنده در حباب 🧠 چت"""
        if text and text.strip():
            state.thinking_log.append({
                "stage": stage, "text": text.strip(), "ts": time.time(),
            })

    # ══════════════════════════════════════════════════════════════
    # فلوی دستی برای پرسش روی فایل‌های آپلودشده
    # (تا این کوئری‌ها هم trace/نمایش زنده/سشن ذخیره‌شده داشته باشند —
    #  قبلاً مسیر فایل هیچ state ای نمی‌ساخت و در Flow دیده نمی‌شد)
    # ══════════════════════════════════════════════════════════════

    def begin_manual_flow(self, query: str, session_id: str = "files") -> AgentState:
        """شروع یک فلوی دستی (خارج از run) — state فعال و trace ساخته می‌شود"""
        state = AgentState(query=query, session_id=session_id)
        state.current_step = FlowStep.START
        state.strategy = "file_analysis"
        self._trace(state, "start", 0.0)
        self.active_state = state
        return state

    def note_file_processed(
        self, state: AgentState, filename: str, report: str, duration: float
    ) -> None:
        """ثبت پردازش یک فایل در trace و فکر (گره FILES در گراف)"""
        state.current_step = FlowStep.START  # هنوز به answer نرسیده‌ایم
        self._trace(state, "files", duration, detail=filename)
        self._think(state, "file", f"**{filename}**:\n{report.strip()[:600]}")

    async def finish_manual_flow(
        self, state: AgentState, answer: str, confidence: float = 0.9
    ) -> AgentState:
        """پایان فلوی دستی: ثبت پاسخ، trace نهایی و ذخیره‌ی سشن"""
        state.final_answer = answer
        state.confidence = confidence
        state.current_step = FlowStep.ANSWER
        self._trace(state, "answer", 0.0, detail=f"confidence={confidence:.2f}")
        self._record_flow(state)
        return state

    # ══════════════════════════════════════════════════════════════
    # Shortcuts — پاسخ مستقیم بدون پایپ‌لاین RAG
    # ══════════════════════════════════════════════════════════════

    _CHAT_SYSTEM = (
        "You are a helpful, friendly assistant. Reply in the same language "
        "as the user (Persian or English). Keep the answer short and natural. "
        "Do not mention retrieval, context, or citations."
    )

    async def _try_shortcut(self, state: AgentState) -> bool:
        """
        کوئری‌های گفت‌وگویی یا محاسباتی ساده را بدون phi3/retrieve/refresh
        جواب می‌دهد. True یعنی state.final_answer پر شد.
        """
        import time as _time
        t0 = _time.perf_counter()

        # «hello»، «مرسی»، «خوبی؟» و ... → پاسخ مستقیم مدل
        if is_conversational(state.query):
            try:
                self._think(state, "shortcut",
                            "کوئری گفت‌وگوی ساده است → پاسخ مستقیم بدون RAG")
                state.final_answer = await self.direct_answer(state.query, state=state)
                state.confidence = 1.0
                state.strategy = "direct_chat"
                state.current_step = FlowStep.ANSWER
                state.timings["direct_chat"] = _time.perf_counter() - t0
                self._trace(state, "shortcut", state.timings["direct_chat"],
                            detail="direct_chat")
                self._trace(state, "answer", 0.0)
                return True
            except Exception as e:
                logger.warning("Direct chat shortcut failed, falling back to full flow: %s", e)
                return False

        # «28% of 8795576» یا «2+2*3» → مستقیم skill محاسبه، بدون planner
        expr = extract_calc_expression(state.query)
        if expr:
            result = await self.skill_executor.execute_skill(
                "calculate", {"expression": expr}
            )
            if result.success and isinstance(result.output, dict) and "result" in result.output:
                state.final_answer = f"{expr} = {result.output['result']}"
                state.confidence = 1.0
                state.strategy = "direct_calculation"
                state.current_step = FlowStep.ANSWER
                state.tool_results = [{
                    "tool": "calculate", "success": True,
                    "output": result.output, "error": None,
                    "execution_time": result.execution_time,
                }]
                state.timings["direct_calculation"] = _time.perf_counter() - t0
                self._trace(state, "shortcut", state.timings["direct_calculation"],
                            detail=f"calculate: {expr}")
                self._trace(state, "answer", 0.0)
                return True
            logger.warning(
                "Calculation shortcut failed for %r (%s) — falling back to full flow",
                expr, result.error,
            )

        return False

    async def direct_answer(
        self,
        question: str,
        context: Optional[str] = None,
        state: Optional[AgentState] = None,
    ) -> str:
        """
        پاسخ مستقیم با مدل answer (granite) بدون retrieval — برای گفت‌وگوی
        ساده و پرسش درباره فایل‌های آپلودشده (context = خروجی skill ها).
        اگر مدل thinking token تولید کند، در state.thinking_log ثبت می‌شود.
        """
        import asyncio as _asyncio

        if "llm_client_factory" not in self.config:
            raise RuntimeError("llm_client_factory not configured")

        async with self.config["llm_client_factory"]() as llm:
            # generate سینک است (پشت execution_slot خود gate) → thread
            resp = await _asyncio.to_thread(
                llm.generate,
                question,
                system=self._CHAT_SYSTEM if context is None else None,
                context=context,
            )
        if state is not None and getattr(resp, "reasoning", None):
            self._think(state, "model_thinking", resp.reasoning)
        return resp.text

    def _record_flow(self, state: AgentState) -> None:
        """ثبت خلاصه اجرای یک query برای dashboard + ذخیره‌ی دائمی سشن"""
        try:
            # مسیر shortcut اصلاً classification ندارد → query_analysis وجود ندارد
            analysis = getattr(state, "query_analysis", None)
            record = {
                "timestamp": time.time(),
                "query": state.query[:200],
                "session_id": state.session_id,
                "query_type": analysis.query_type.value if analysis else "unknown",
                "complexity": analysis.complexity.value if analysis else "unknown",
                "strategy": getattr(analysis, "suggested_strategy", "") if analysis else "",
                "final_step": state.current_step.value,
                "confidence": state.confidence,
                "iterations": state.iteration,
                "refresh_attempts": state.refresh_attempts,
                "chunks_retrieved": len(state.retrieval.vector_chunks) if state.retrieval else 0,
                "graph_nodes": len(state.retrieval.graph_nodes) if state.retrieval else 0,
                "timings": dict(state.timings),
                "errors": list(state.errors),
                "answered": bool(state.final_answer) and state.current_step == FlowStep.ANSWER,
                "answer": (state.final_answer or "")[:1500],
                "trace": [dict(e) for e in state.flow_trace],
            }
            self.flow_history.append(record)
            if len(self.flow_history) > 200:
                self.flow_history.pop(0)
            # ذخیره‌ی دائمی — بعد از restart هم در tab «Flow» قابل مرور است
            self._session_store.append(record)
        except Exception as e:
            logger.debug("flow history record failed: %s", e)
    
    # ══════════════════════════════════════════════════════════════
    # UNDERSTAND + Analysis
    # ══════════════════════════════════════════════════════════════
    
    async def _step_understand_and_analyze(self, state: AgentState) -> None:
        """
        مرحله UNDERSTAND با تحلیل پیشرفته query
        """
        t0 = time.perf_counter()
        
        # 1. Query Classification
        query_analysis = self.query_classifier.classify(state.query)
        
        logger.info(
            f"Query classified: type={query_analysis.query_type.value}, "
            f"complexity={query_analysis.complexity.value}, "
            f"strategy={query_analysis.suggested_strategy}"
        )
        self._think(
            state, "classify",
            f"نوع کوئری: {query_analysis.query_type.value} | "
            f"پیچیدگی: {query_analysis.complexity.value} | "
            f"استراتژی: {query_analysis.suggested_strategy}",
        )
        
        # 2. فهم پایه با Phi4 (اگر موجود باشد)
        # برای کوئری‌های ساده classifier به‌تنهایی کافی است — اجرای phi4
        # (~۴۵ ثانیه روی MLX) فقط entities/sub-question تکراری تولید می‌کرد.
        # استثنا: اگر کوئری به فایل media اشاره دارد، phi4 لازم است تا
        # tool مناسب (YOLO/Whisper/...) را انتخاب کند.
        skip_phi4 = (
            query_analysis.complexity == QueryComplexity.LOW
            and query_analysis.query_type in {
                QueryType.SIMPLE_FACT, QueryType.CREATIVE, QueryType.CALCULATION,
            }
            and not self._MEDIA_HINT.search(state.query)
        )
        if skip_phi4:
            logger.info(
                "Skipping phi4 understanding (type=%s, complexity=low) — "
                "classifier plan is sufficient",
                query_analysis.query_type.value,
            )
        if "phi4_mini_factory" in self.config and not skip_phi4:
            async with self.config["phi4_mini_factory"]() as phi4:
                result = await phi4.understand_and_plan(
                    query=state.query,
                    history=state.history,
                )
                
                state.intent = result["intent"]
                state.tool_calls = result["tool_calls"]
                state.sub_questions = result["sub_questions"]
                state.extracted_entities = result["entities"]
                state.plan_steps = result["steps"]
                state.strategy = result["strategy"]

                plan_lines = [
                    f"{s.get('id')}. {s.get('action')}: {s.get('description', '')[:80]}"
                    for s in (result.get("steps") or [])
                ]
                self._think(
                    state, "plan",
                    f"intent: {result['intent']} | strategy: {result['strategy']}"
                    + (f"\nزیرسوال‌ها: {result['sub_questions']}"
                       if result.get("sub_questions") else "")
                    + ("\nبرنامه:\n" + "\n".join(plan_lines) if plan_lines else ""),
                )
        
        # 3. اجرای tool calls — only execute non-retrieval tools (calculate,
        #    summarize, etc.) since retrieval is handled by the dynamic plan
        _retrieval_tools = {"search", "web_search", "retrieve", "graph_query"}
        if state.tool_calls:
            actionable_calls = [
                tc for tc in state.tool_calls
                if tc.get("tool") not in _retrieval_tools
            ]
            if actionable_calls:
                logger.info(f"Executing {len(actionable_calls)} tool calls from phi4")
                original_calls = state.tool_calls
                state.tool_calls = actionable_calls
                await self._execute_tool_calls(state)
                state.tool_calls = original_calls
            else:
                logger.info(
                    f"Skipping {len(state.tool_calls)} retrieval-type tool calls "
                    f"(handled by dynamic plan)"
                )

        # 3.5. If tool calls already answered the query, short-circuit
        if self._tool_results_answer_query(state, query_analysis):
            state.query_analysis = query_analysis
            state.dynamic_plan = DynamicPlan(
                required_steps=[], skip_steps=[], max_iterations=0
            )
            state.timings["understand"] = time.perf_counter() - t0
            self._trace(state, "understand", state.timings["understand"],
                        detail="tool call answered the query")
            state.current_step = FlowStep.ANSWER
            return

        # 4. اجرای plan steps — skip retrieval-type steps since the dynamic
        #    plan handles RETRIEVE/REFRESH via the real agents
        _retrieval_actions = {"retrieve", "search", "web_search"}
        actionable_steps = [
            s for s in (state.plan_steps or [])
            if s.get("action") not in _retrieval_actions
            and s.get("tool") not in _retrieval_actions
        ]
        if actionable_steps and state.strategy in ["sequential", "parallel", "iterative"]:
            logger.info(f"Executing {len(actionable_steps)} non-retrieval plan steps from phi4")
            state.plan_steps = actionable_steps
            await self._execute_plan_steps(state)

        # 5. تولید Dynamic Plan
        dynamic_plan = self._generate_dynamic_plan(query_analysis, state)
        state.dynamic_plan = dynamic_plan

        # ذخیره query_analysis در state
        state.query_analysis = query_analysis

        state.timings["understand"] = time.perf_counter() - t0
        self._trace(
            state, "understand", state.timings["understand"],
            detail=f"type={query_analysis.query_type.value}"
                   f"{' (phi4 skipped)' if skip_phi4 else ''}",
        )

        # Set initial step to the first required step in the dynamic plan
        if dynamic_plan.required_steps:
            state.current_step = dynamic_plan.required_steps[0]
        else:
            state.current_step = FlowStep.RETRIEVE

        # If plan requires refresh, set the flag so refresh agent acts on it
        if dynamic_plan.force_refresh:
            state.refresh_needed = True
    
    # ══════════════════════════════════════════════════════════════
    # Dynamic Plan Generation
    # ══════════════════════════════════════════════════════════════
    
    def _generate_dynamic_plan(
        self,
        analysis: QueryAnalysis,
        state: AgentState,
    ) -> DynamicPlan:
        """تولید plan بر اساس نوع query"""
        
        plan = DynamicPlan()
        
        # بر اساس query_type
        if analysis.query_type == QueryType.SIMPLE_FACT:
            # سوال ساده: سریع و مستقیم
            plan.required_steps = [FlowStep.RETRIEVE, FlowStep.REASON]
            plan.skip_steps = [FlowStep.ASSESS, FlowStep.REFRESH, FlowStep.VALIDATE]
            plan.max_iterations = 1
            plan.allow_early_exit = True
            plan.confidence_threshold = 0.7
            
        elif analysis.query_type == QueryType.TEMPORAL:
            # سوال زمانی: حتما refresh
            plan.required_steps = [FlowStep.REFRESH, FlowStep.RETRIEVE, FlowStep.REASON, FlowStep.VALIDATE]
            plan.skip_steps = []
            plan.force_refresh = True
            plan.max_iterations = 5
            
        elif analysis.query_type == QueryType.MULTI_HOP:
            # چند مرحله‌ای: تکرار retrieve
            plan.required_steps = [FlowStep.RETRIEVE, FlowStep.ASSESS, FlowStep.REASON, FlowStep.VALIDATE]
            plan.skip_steps = []
            plan.max_retrieve_iterations = 3
            plan.enable_sub_questions = True
            
        elif analysis.query_type == QueryType.COMPLEX_REASONING:
            # استدلال پیچیده: همه مراحل
            plan.required_steps = [
                FlowStep.RETRIEVE,
                FlowStep.ASSESS,
                FlowStep.REASON,
                FlowStep.VALIDATE,
            ]
            plan.skip_steps = []
            plan.max_iterations = 3
            
        elif analysis.query_type == QueryType.CALCULATION:
            plan.required_steps = [FlowStep.REASON]
            plan.skip_steps = [FlowStep.RETRIEVE, FlowStep.ASSESS, FlowStep.REFRESH, FlowStep.VALIDATE]
            plan.max_iterations = 1
            plan.allow_early_exit = True
            plan.use_simple_answer = True
            plan.disable_graph_traversal = True

        elif analysis.query_type == QueryType.CREATIVE:
            # خلاقانه: کم‌تر retrieval
            plan.required_steps = [FlowStep.REASON]
            plan.skip_steps = [FlowStep.RETRIEVE, FlowStep.ASSESS, FlowStep.REFRESH, FlowStep.VALIDATE]
            plan.max_iterations = 1
            plan.use_simple_answer = True
            plan.disable_graph_traversal = True
            
        elif analysis.query_type == QueryType.COMPARATIVE:
            # مقایسه‌ای: نیاز به retrieve دقیق
            plan.required_steps = [FlowStep.RETRIEVE, FlowStep.REASON, FlowStep.VALIDATE]
            plan.skip_steps = [FlowStep.ASSESS]
            plan.max_iterations = 2
            
        else:
            # Default: hybrid
            plan.required_steps = [FlowStep.RETRIEVE, FlowStep.ASSESS, FlowStep.REASON, FlowStep.VALIDATE]
            plan.skip_steps = []
        
        # تعدیل بر اساس complexity
        if analysis.complexity == QueryComplexity.LOW:
            plan.max_iterations = 1
            plan.allow_early_exit = True
        elif analysis.complexity == QueryComplexity.HIGH:
            plan.max_iterations = 3
        
        # تعدیل بر اساس نیازمندی‌ها
        if not analysis.requires_graph:
            plan.disable_graph_traversal = True
        
        if analysis.requires_refresh:
            plan.force_refresh = True
        
        logger.debug(f"Generated plan: required={[s.value for s in plan.required_steps]}, skip={[s.value for s in plan.skip_steps]}")
        
        return plan
    
    # ══════════════════════════════════════════════════════════════
    # Dynamic Plan Execution
    # ══════════════════════════════════════════════════════════════
    
    async def _execute_dynamic_plan(self, state: AgentState) -> None:
        """اجرای plan داینامیک"""
        plan: DynamicPlan = state.dynamic_plan

        # iteration = تعداد چرخه‌ها (بازگشت به step قبلاً اجرا شده)،
        # نه تعداد step ها — وگرنه فلوی ۵ مرحله‌ای با max_iterations=2 وسط راه قطع می‌شود
        executed_steps: set[FlowStep] = set()

        while state.current_step not in (FlowStep.ANSWER, FlowStep.ERROR):
            if state.iteration >= plan.max_iterations:
                logger.warning("Max iterations reached — finalizing with best-effort answer")
                await self._finalize_best_effort(state, plan)
                break

            # بررسی early exit
            if plan.allow_early_exit and self._should_early_exit(state, plan):
                logger.info("Early exit triggered")
                state.current_step = FlowStep.ANSWER
                break

            # اجرای step فعلی
            current = state.current_step
            next_step = await self._execute_step(state, plan)
            executed_steps.add(current)

            if next_step is None:
                next_step = FlowStep.ANSWER

            # بازگشت به step قبلی = یک چرخه کامل
            if next_step in executed_steps:
                state.iteration += 1
                executed_steps.clear()

            self._transition(state, next_step)

        # هیچ‌وقت بدون پاسخ خارج نشو (UI فقط final_answer را نشان می‌دهد)
        if state.current_step == FlowStep.ANSWER and not state.final_answer:
            await self._finalize_best_effort(state, plan)
    
    async def _execute_step(
        self,
        state: AgentState,
        plan: DynamicPlan,
    ) -> Optional[FlowStep]:
        """اجرای یک step"""
        step = state.current_step
        
        # بررسی skip
        if step in plan.skip_steps:
            logger.debug(f"Skipping step: {step.value}")
            return self._get_next_required_step(step, plan)
        
        t0 = time.perf_counter()

        try:
            # جدول گره‌ها (نگاشت node → handler) — معادل add_node در LangGraph
            handler = self._STEP_HANDLERS.get(step)
            if handler is None:
                next_step = FlowStep.ERROR
            else:
                next_step = await handler(self, state, plan)


        except Exception as e:
            logger.error(f"Step {step} failed: {e}", exc_info=True)
            state.errors.append(f"[{step}] {str(e)}")
            next_step = FlowStep.ERROR
        finally:
            gc.collect()
            self._release_gpu_cache()

        state.timings[step.value] = time.perf_counter() - t0
        self._trace(
            state, step.value, state.timings[step.value],
            status="error" if next_step == FlowStep.ERROR else "done",
            detail=self._step_detail(state, step),
        )
        return next_step

    @staticmethod
    def _step_detail(state: AgentState, step: FlowStep) -> str:
        """خلاصه‌ی یک‌خطی نتیجه‌ی step برای نمایش در گراف فلو"""
        try:
            if step == FlowStep.RETRIEVE and state.retrieval:
                return (
                    f"chunks={len(state.retrieval.vector_chunks)} "
                    f"conf={state.retrieval.confidence:.2f}"
                )
            if step == FlowStep.ASSESS and state.retrieval:
                return f"conf={state.retrieval.confidence:.2f}"
            if step == FlowStep.REFRESH:
                m = state.metadata.get("last_refresh_metrics") or {}
                return f"items={m.get('total_items', 0)} chunks={m.get('chunks_ingested', 0)}"
            if step in (FlowStep.REASON, FlowStep.VALIDATE):
                return f"conf={state.confidence:.2f}"
        except Exception:
            pass
        return ""

    @staticmethod
    def _release_gpu_cache() -> None:
        """آزادسازی کش Metal بین step ها — جلوگیری از OOM وقتی چند مدل MLX لود می‌شوند"""
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
    # ══════════════════════════════════════════════════════════════
    # Tool/Skill Call Execution
    # ══════════════════════════════════════════════════════════════
    
    async def _execute_tool_calls(self, state: AgentState) -> None:
        """
        اجرای tool calls که در phi4.understand_and_plan تشخیص داده شده
        
        این متد tool_calls را از state می‌گیرد و اجرا می‌کند
        """
        if not state.tool_calls:
            logger.debug("No tool calls to execute")
            return
        
        logger.info(f"Executing {len(state.tool_calls)} tool calls")
        
        # اجرای tool calls with query context for auto-filling missing args
        context = {"query": state.query}
        results = await self.skill_executor.execute_multiple_skills(
            state.tool_calls,
            parallel=False,  # ترتیبی برای حفظ dependencies
            context=context
        )
        
        # ذخیره نتایج در state
        state.tool_results = []
        for i, result in enumerate(results):
            tool_call = state.tool_calls[i]
            state.tool_results.append({
                "tool": tool_call.get("tool"),
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "execution_time": result.execution_time
            })
            
            if result.success:
                logger.info(f"Tool '{tool_call.get('tool')}' executed successfully in {result.execution_time:.2f}s")
            else:
                logger.error(f"Tool '{tool_call.get('tool')}' failed: {result.error}")
    
    async def _execute_plan_steps(self, state: AgentState) -> None:
        """
        اجرای steps که در phi4.understand_and_plan تولید شده
        
        این متد plan_steps را از state می‌گیرد و به ترتیب اجرا می‌کند
        """
        if not state.plan_steps:
            logger.debug("No plan steps to execute")
            return
        
        logger.info(f"Executing {len(state.plan_steps)} plan steps")
        
        # نگاشت step id به نتیجه
        step_results = {}
        
        for step in state.plan_steps:
            step_id = step.get("id")
            action = step.get("action")
            depends_on = step.get("depends_on", [])
            tool = step.get("tool")
            args = step.get("args", {})
            
            logger.info(f"Executing step {step_id}: {action}")

            # هر step باید query خودش را بگیرد (refined از planner)،
            # نه query خام کاربر — وگرنه همه tool ها پیام یکسان و خام می‌گیرند
            step_query = (
                args.get("query")
                or step.get("description")
                or state.query
            )

            # بررسی dependencies (normalize types for comparison)
            if depends_on:
                result_keys = {str(k) for k in step_results}
                missing_deps = [dep for dep in depends_on if str(dep) not in result_keys]
                if missing_deps:
                    logger.error(f"Step {step_id} has missing dependencies: {missing_deps}")
                    step_results[step_id] = {
                        "success": False,
                        "error": f"Missing dependencies: {missing_deps}"
                    }
                    continue
            
            # اجرای step
            try:
                skill_context = {"query": step_query}
                if tool:
                    result = await self.skill_executor.execute_skill(tool, args, skill_context)
                    if not result.success and isinstance(result.error, str) and "missing" in result.error.lower():
                        # tool called with wrong/missing args — fall through to action handler
                        logger.warning(
                            "Step %s: tool '%s' rejected args %s (%s), falling back to action '%s'",
                            step_id, tool, args, result.error, action
                        )
                        result = await self._execute_action(
                            action, state, {"query": step_query}, step_results
                        )
                        step_results[step_id] = result
                    else:
                        step_results[step_id] = {
                            "success": result.success,
                            "output": result.output,
                            "error": result.error,
                        }
                else:
                    # اگر tool نیست، بر اساس action اجرا کن
                    result = await self._execute_action(action, state, args, step_results)
                    step_results[step_id] = result
                
                if step_results[step_id]["success"]:
                    logger.info(f"Step {step_id} completed successfully")
                else:
                    logger.error(f"Step {step_id} failed: {step_results[step_id].get('error')}")
                    
            except Exception as e:
                logger.error(f"Step {step_id} execution failed: {e}", exc_info=True)
                step_results[step_id] = {
                    "success": False,
                    "error": str(e)
                }
        
        # ذخیره نتایج در state
        state.step_results = step_results
    
    async def _execute_action(
        self,
        action: str,
        state: AgentState,
        args: dict,
        previous_results: dict
    ) -> dict:
        """
        اجرای یک action (retrieve, reason, search, etc.)
        
        Args:
            action: نوع action
            state: state فعلی
            args: آرگومان‌های action
            previous_results: نتایج steps قبلی
            
        Returns:
            dict با success, output, error
        """
        try:
            if action == "retrieve":
                # اجرای retrieve
                await self.retriever.startup()
                await self.retriever.retrieve(state)
                return {
                    "success": True,
                    "output": {
                        "chunks": len(state.retrieval.vector_chunks),
                        "confidence": state.retrieval.confidence
                    }
                }
            
            elif action == "search":
                # اجرای search (از skill executor)
                query = args.get("query", state.query)
                result = await self.skill_executor.execute_skill("search", {"query": query})
                return {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error
                }
            
            elif action == "reason":
                # اجرای reasoning
                # این باید با reasoner agent انجام شود
                return {
                    "success": True,
                    "output": {"reasoning": "Reasoning completed"}
                }
            
            elif action == "summarize":
                text = args.get("text", "")
                if not text:
                    chunks = state.retrieval.vector_chunks[:5] if state.retrieval else []
                    text = "\n".join(ch.get("content", "") for ch in chunks)
                if text:
                    result = await self.skill_executor.execute_skill(
                        "summarize", {"text": text}, {"query": state.query}
                    )
                    return {
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                    }
                return {
                    "success": True,
                    "output": {"summary": "No text available to summarize"}
                }
            
            elif action == "compare":
                # مقایسه
                return {
                    "success": True,
                    "output": {"comparison": "Comparison completed"}
                }
            
            elif action == "validate":
                # اعتبارسنجی
                await self.validator.validate(state)
                return {
                    "success": True,
                    "output": {
                        "valid": state.validation.is_valid,
                        "confidence": state.validation.confidence
                    }
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
                
        except Exception as e:
            logger.error(f"Action '{action}' failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    
    # ══════════════════════════════════════════════════════════════
    # Step Handlers
    # ══════════════════════════════════════════════════════════════
    
    async def _step_retrieve(self, state: AgentState, plan: DynamicPlan) -> FlowStep:
        """مرحله RETRIEVE"""
        if not self.retriever._ready:
            try:
                await self.retriever.startup()
            except Exception as e:
                logger.error("Retriever startup failed: %s", e)
                state.errors.append(f"retriever_unavailable: {e}")
                return FlowStep.REASON
        await self.retriever.retrieve(state)
        
        # اگر skip_assessment، مستقیم به REASON
        if plan.skip_assessment:
            return FlowStep.REASON
        
        return FlowStep.ASSESS
    
    async def _step_assess(self, state: AgentState, plan: DynamicPlan) -> FlowStep:
        """مرحله ASSESS"""
        confidence = state.retrieval.confidence

        # refresh فقط تا سقف مجاز — وگرنه فلوی temporal در حلقه
        # ASSESS→REFRESH→RETRIEVE→ASSESS گیر می‌کند و هیچ‌وقت به REASON نمی‌رسد
        can_refresh = self._can_refresh(state, plan)

        # بررسی confidence
        if confidence < self.config.get("confidence_threshold", 0.65) and can_refresh:
            state.refresh_needed = True
            return FlowStep.REFRESH

        # Gap detection (اگر Phi4 موجود باشد)
        if "phi4_mini_factory" in self.config and can_refresh:
            fused_context = "\n".join(
                ch.get("content", "") for ch in state.retrieval.vector_chunks[:5]
            )

            async with self.config["phi4_mini_factory"]() as phi4:
                gap = await phi4.detect_gaps(state.query, fused_context)

            if not gap.get("sufficient", True) and gap.get("confidence", 1.0) < 0.7:
                state.refresh_needed = True
                return FlowStep.REFRESH

        return FlowStep.REASON

    async def _step_refresh(self, state: AgentState, plan: DynamicPlan) -> FlowStep:
        """مرحله REFRESH"""
        state.refresh_needed = True
        state.refresh_attempts += 1
        if not self.refresher._ready:
            await self.refresher.startup()
        await self.refresher.refresh(state)
        return FlowStep.RETRIEVE
    
    async def _step_reason(self, state: AgentState, plan: DynamicPlan) -> FlowStep:
        """مرحله REASON"""
        if "llm_client_factory" in self.config:
            async with self.config["llm_client_factory"]() as reasoning_model:
                reasoner = ReasonerAgent(self.config)
                await reasoner.reason(state)
        
        # اگر use_simple_answer، مستقیم به ANSWER
        if plan.use_simple_answer:
            return FlowStep.ANSWER
        
        # اگر skip VALIDATE
        if FlowStep.VALIDATE in plan.skip_steps:
            return FlowStep.ANSWER
        
        return FlowStep.VALIDATE
    
    async def _step_validate(self, state: AgentState, plan: DynamicPlan) -> FlowStep:
        """مرحله VALIDATE"""
        result = await self.validator.validate(state)

        # همیشه بهترین پاسخ فعلی را نگه دار تا خروج در هر نقطه‌ای پاسخ داشته باشد
        state.final_answer = result["answer"] or state.final_answer
        state.confidence = result["score"]

        if result["score"] >= plan.confidence_threshold:
            return FlowStep.ANSWER
        if state.iteration >= plan.max_iterations:
            return FlowStep.ANSWER
        if result["score"] >= 0.50:
            return FlowStep.RETRIEVE
        if self._can_refresh(state, plan):
            return FlowStep.REFRESH
        # نه refresh مجاز است نه پاسخ بهتری در دسترس — با همین پاسخ خارج شو
        return FlowStep.ANSWER

    # ══════════════════════════════════════════════════════════════
    # جدول گره‌های گراف — معادل add_node در LangGraph
    # (هر handler نام گره‌ی بعدی را برمی‌گرداند = conditional edge)
    # ══════════════════════════════════════════════════════════════
    _STEP_HANDLERS = {
        FlowStep.RETRIEVE: _step_retrieve,
        FlowStep.ASSESS:   _step_assess,
        FlowStep.REFRESH:  _step_refresh,
        FlowStep.REASON:   _step_reason,
        FlowStep.VALIDATE: _step_validate,
    }


    # ══════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════
    
    # اشاره به فایل media در متن کوئری → phi4 برای انتخاب tool لازم است
    _MEDIA_HINT = re.compile(
        r"\.(png|jpe?g|bmp|webp|gif|tiff|wav|mp3|m4a|flac|ogg|aac|pdf|docx?|xlsx?|csv)\b",
        re.IGNORECASE,
    )

    # tool های media که خروجی‌شان مستقیم قابل ارائه به کاربر است
    _MEDIA_TOOLS = {
        "detect_objects_in_image", "estimate_pose_in_image",
        "segment_image_objects", "transcribe_audio_file", "list_local_models",
    }

    def _tool_results_answer_query(
        self, state: AgentState, analysis: QueryAnalysis
    ) -> bool:
        """Check if tool results already answer the query (calculate / media tools)."""
        if not state.tool_results:
            return False

        successful = [tr for tr in state.tool_results if tr.get("success")]
        if not successful:
            return False

        media_calls = any(tr.get("tool") in self._MEDIA_TOOLS for tr in successful)
        if analysis.query_type not in (QueryType.CALCULATION,) and not media_calls:
            return False

        parts = []
        for tr in successful:
            output = tr.get("output", {})
            if not isinstance(output, dict):
                continue
            if "result" in output:
                expr = output.get("expression", "")
                result = output["result"]
                parts.append(f"{expr} = {result}" if expr else str(result))
            elif "summary" in output:
                parts.append(output["summary"])
            elif output.get("objects") is not None and output.get("success"):
                objs = output["objects"]
                names = ", ".join(
                    f"{o.get('class_name', o.get('label', '?'))}"
                    f" ({o.get('confidence', 0):.0%})" if isinstance(o, dict) else str(o)
                    for o in objs[:15]
                )
                parts.append(
                    f"اشیاء تشخیص‌داده‌شده ({output.get('count', len(objs))}): {names}"
                    if objs else "شیء مشخصی در تصویر پیدا نشد."
                )
            elif output.get("text") and output.get("success"):
                lang = output.get("language", "?")
                parts.append(f"متن پیاده‌شده (زبان={lang}):\n{output['text']}")
            elif output.get("persons") is not None and output.get("success"):
                parts.append(f"تعداد افراد با ژست تشخیص‌داده‌شده: {output.get('count', 0)}")
            elif output.get("models") is not None and output.get("success"):
                lines = [
                    f"- {m['name']} [{m['kind']}]: {', '.join(m['capabilities'])}"
                    f"{'' if m['available'] else ' (روی دیسک موجود نیست)'}"
                    for m in output["models"]
                ]
                parts.append("مدل‌های لوکال:\n" + "\n".join(lines))

        if parts:
            state.final_answer = "\n\n".join(parts)
            state.confidence = 1.0 if not media_calls else 0.9
            logger.info("Tool results answered query directly (%d parts)", len(parts))
            return True

        return False

    def _can_refresh(self, state: AgentState, plan: DynamicPlan) -> bool:
        """آیا هنوز بودجه refresh باقی مانده است؟"""
        budget = max(1, plan.max_retrieve_iterations)
        return state.refresh_attempts < budget

    async def _finalize_best_effort(self, state: AgentState, plan: DynamicPlan) -> None:
        """
        تولید بهترین پاسخ ممکن قبل از خروج — فلو هیچ‌وقت نباید بدون
        final_answer تمام شود (UI فقط همین فیلد را نشان می‌دهد)
        """
        if not state.final_answer:
            try:
                await self._step_reason(state, plan)
            except Exception as e:
                logger.error("Best-effort reasoning failed: %s", e, exc_info=True)
                state.errors.append(f"finalize_reason: {e}")

        if not state.final_answer:
            if state.errors:
                state.final_answer = (
                    "متأسفانه پردازش این سوال با خطا مواجه شد و پاسخی تولید نشد. "
                    f"(آخرین خطا: {state.errors[-1]})"
                )
            else:
                state.final_answer = (
                    "اطلاعات کافی برای پاسخ به این سوال در دانش فعلی سیستم یافت نشد. "
                    "لطفاً سوال را دقیق‌تر مطرح کنید یا منابع جدیدی اضافه کنید."
                )
            state.confidence = 0.0

        state.current_step = FlowStep.ANSWER

    def _should_early_exit(self, state: AgentState, plan: DynamicPlan) -> bool:
        """آیا می‌توانیم زودتر خارج شویم؟"""
        # confidence بالا
        if state.confidence >= 0.9:
            return True
        
        # پاسخ موجود با confidence قابل قبول
        if state.final_answer and state.confidence >= plan.confidence_threshold:
            return True
        
        return False
    
    def _get_next_required_step(
        self,
        current_step: FlowStep,
        plan: DynamicPlan,
    ) -> Optional[FlowStep]:
        """پیدا کردن step بعدی required"""
        # ترتیب استاندارد
        standard_order = [
            FlowStep.RETRIEVE,
            FlowStep.ASSESS,
            FlowStep.REFRESH,
            FlowStep.REASON,
            FlowStep.VALIDATE,
            FlowStep.ANSWER,
        ]
        
        try:
            current_idx = standard_order.index(current_step)
            for step in standard_order[current_idx + 1:]:
                if step in plan.required_steps and step not in plan.skip_steps:
                    return step
        except ValueError:
            pass
        
        return FlowStep.ANSWER
    
    def _transition(self, state: AgentState, next_step: FlowStep) -> None:
        """انتقال به step بعدی"""
        allowed = self._transitions.get(state.current_step, [])
        
        if next_step not in allowed and next_step != FlowStep.ERROR:
            logger.warning(
                f"Invalid transition {state.current_step} -> {next_step}, "
                f"allowed: {allowed}"
            )
            # در حالت adaptive، اجازه می‌دهیم
            # state.current_step = FlowStep.ERROR
            # return
        
        state.current_step = next_step
    
    # ══════════════════════════════════════════════════════════════
    # Fallback: Fixed Flow
    # ══════════════════════════════════════════════════════════════
    
    async def _execute_fixed_flow(self, state: AgentState) -> None:
        """فلوی ثابت به عنوان fallback"""
        logger.info("Using fixed flow (fallback)")
        
        while state.current_step not in (FlowStep.ANSWER, FlowStep.ERROR):
            if state.iteration > state.max_iterations:
                logger.warning("Max iterations reached")
                state.current_step = FlowStep.ANSWER
                break
            
            next_step = await self._execute_fixed_step(state)
            self._transition(state, next_step)
            state.iteration += 1
    
    async def _execute_fixed_step(self, state: AgentState) -> FlowStep:
        """اجرای step در فلوی ثابت"""
        step = state.current_step
        
        if step == FlowStep.RETRIEVE:
            await self.retriever.startup()
            await self.retriever.retrieve(state)
            return FlowStep.ASSESS
        
        elif step == FlowStep.ASSESS:
            c = state.retrieval.confidence
            if c < self.config.get("confidence_threshold", 0.65):
                state.refresh_needed = True
                return FlowStep.REFRESH
            return FlowStep.REASON
        
        elif step == FlowStep.REFRESH:
            await self.refresher.startup()
            await self.refresher.refresh(state)
            return FlowStep.RETRIEVE
        
        elif step == FlowStep.REASON:
            if "llm_client_factory" in self.config:
                async with self.config["llm_client_factory"]() as reasoning_model:
                    reasoner = ReasonerAgent(self.config)
                    await reasoner.reason(state)
            return FlowStep.VALIDATE
        
        elif step == FlowStep.VALIDATE:
            result = await self.validator.validate(state)
            if result["score"] >= 0.75:
                state.final_answer = result["answer"]
                state.confidence = result["score"]
                return FlowStep.ANSWER
            elif result["score"] >= 0.50:
                return FlowStep.RETRIEVE
            else:
                return FlowStep.REFRESH
        
        return FlowStep.ERROR

