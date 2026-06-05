import time
import logging
import gc
from agents.state import AgentState, FlowStep
from agents.retriever_agent import RetrieverAgent
from agents.reasoner_agent import ReasonerAgent
from agents.validator_agent import ValidatorAgent
from agents.knowledge_refresh_agent import KnowledgeRefreshAgent
from agents.tool_registry import register_tool

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator با قابلیت Lazy-Loading و Auto-Unload.
    مدل‌ها فقط در زمان نیاز لود شده و بلافاصله پس از اجرا از حافظه خارج می‌شوند.
    """

    def __init__(self, config: dict):
        self.config = config
        
        # ایجنت‌های سبک (بدون مدل سنگین در بدنه اصلی) را می‌توان اینجا نگه داشت
        # اما اگر خود این ایجنت‌ها مدل لود می‌کنند، باید داخل متدها ساخته شوند
        self.retriever = RetrieverAgent(config)
        self.validator = ValidatorAgent(config)
        self.refresher = KnowledgeRefreshAgent(config)

        # نگاشت انتقال‌ها
        self._transitions: dict[FlowStep, list[FlowStep]] = {
            FlowStep.START:      [FlowStep.UNDERSTAND],
            FlowStep.UNDERSTAND: [FlowStep.RETRIEVE, FlowStep.ERROR],
            FlowStep.RETRIEVE:   [FlowStep.ASSESS],
            FlowStep.ASSESS:     [FlowStep.REASON, FlowStep.REFRESH],
            FlowStep.REFRESH:    [FlowStep.RETRIEVE],
            FlowStep.REASON:     [FlowStep.VALIDATE],
            FlowStep.VALIDATE:   [FlowStep.ANSWER, FlowStep.RETRIEVE, FlowStep.REFRESH],
            FlowStep.ANSWER:     [],
            FlowStep.ERROR:      [],
        }

    # ─────────────────────────────────────────────
    # TOOLS (Lazy Version)
    # ─────────────────────────────────────────────

    @register_tool("retrieve")
    async def tool_retrieve(self, state: AgentState, query: str = "", **_) -> None:
        q = query or state.query
        state.retrieval = await self.retriever.retrieve(state, q)
    
    @register_tool("detect_gaps")
    async def tool_detect_gaps(self, state: AgentState, **_) -> None:
        # لود موقت Phi4 برای تشخیص Gap
        async with self.config["phi4_mini_factory"]() as phi4:
            gaps = await phi4.detect_gaps(state.query, state.retrieval.context)
            state.gaps = gaps
            if not gaps.get("sufficient", True):
                state.refresh_needed = True
    
    @register_tool("reason")
    async def tool_reason(self, state: AgentState, **_) -> None:
        # لود موقت Phi4 یا Granite برای پاسخ‌دهی
        async with self.config["phi4_mini_factory"]() as phi4:
            state.answer = await phi4.reason(state.query, state.retrieval.context)
    
    @register_tool("sub_question")
    async def tool_sub_question(self, state: AgentState, description: str = "", **_) -> None:
        result = await self.retriever.retrieve(state, description)
        if state.retrieval:
            state.retrieval.merge(result)
        else:
            state.retrieval = result

    # ─────────────────────────────────────────────
    # EXECUTION CORE
    # ─────────────────────────────────────────────

    async def run(self, query: str, session_id: str) -> AgentState:
        state = AgentState(query=query, session_id=session_id)
        state.current_step = FlowStep.START

        while state.current_step not in (FlowStep.ANSWER, FlowStep.ERROR):
            if state.iteration > state.max_iterations:
                logger.warning("Max iterations reached")
                state.current_step = FlowStep.ANSWER
                
            next_step = await self._execute_step(state)
            self._transition(state, next_step)
 
        gc.collect()
        logger.info(state)
        return state.final_answer

    async def _execute_step(self, state: AgentState) -> FlowStep:
        step = state.current_step
        t0 = time.perf_counter()
        logger.info(f"[{state.session_id}] Executing step: {step}")

        try:
            handlers = {
                FlowStep.START:      self._handle_start,
                FlowStep.UNDERSTAND: self._step_understand,
                FlowStep.RETRIEVE:   self._step_retrieve,
                FlowStep.ASSESS:     self._step_assess,
                FlowStep.REFRESH:    self._step_refresh,
                FlowStep.REASON:     self._step_reason,
                FlowStep.VALIDATE:   self._step_validate,
                FlowStep.ANSWER:     {} # بازگشت برای نهایی سازی
            }
            
            handler = handlers.get(step)
            if handler:
                next_step = await handler(state)
            else:
                next_step = FlowStep.ERROR

        except Exception as e:
            state.errors.append(f"[{step}] {str(e)}")
            logger.error(f"Step {step} failed: {e}", exc_info=True)
            next_step = FlowStep.ERROR
        finally:
            # یک لایه اطمینان برای آزاد سازی حافظه در هر مرحله
            gc.collect()

        state.timings[step.value] = time.perf_counter() - t0
        return next_step

    # ─────────────────────────────────────────────
    # STEP HANDLERS (With Context Managers)
    # ─────────────────────────────────────────────

    async def _handle_start(self, state: AgentState) -> FlowStep:
        return FlowStep.UNDERSTAND

    async def _step_understand(self, state: AgentState) -> FlowStep:
        # مدل لود می‌شود -> اجرا می‌شود -> بلافاصله Unload می‌شود
        async with self.config["phi4_mini_factory"]() as phi4:
            result = await phi4.understand_and_plan(
                query=state.query,
                history=state.history,
            )

        logger.info(result)
        state.intent             = result["intent"]
        state.tool_calls         = result["tool_calls"]
        state.sub_questions      = result["sub_questions"]
        state.extracted_entities = result["entities"]
        state.plan_steps         = result["steps"]
        state.strategy           = result["strategy"]
    
        return FlowStep.RETRIEVE

    async def _step_retrieve(self, state: AgentState) -> FlowStep:
        await self.retriever.startup()
        # اگر RetrieverAgent از مدل Embedding داخلی استفاده می‌کند، 
        # باید در پایان متد retrieve آن را ببندد.
        state = await self.retriever.retrieve(state )#, state.query)
        return FlowStep.ASSESS

    async def _step_assess(self, state: AgentState) -> FlowStep:
        # 1. چک کردن Confidence (بدون LLM)
        c = state.retrieval.confidence
        if c < self.config.get("confidence_threshold", 0.65) and state.iteration < state.max_iterations:
            state.refresh_needed = True
            return FlowStep.REFRESH
    
        # 2. تشخیص Gap تخصصی (با لود موقت LLM)
        fused_context = "\n".join(
            ch.get("content", "") for ch in state.retrieval.vector_chunks[:5]
        )
        
        if  state.iteration < state.max_iterations:
            async with self.config["phi4_mini_factory"]() as phi4:
               gap = await phi4.detect_gaps(state.query, fused_context)
    
            if not gap.get("sufficient", True) and gap.get("confidence", 1.0) < 0.7 and state.iteration < state.max_iterations:
               state.refresh_queries = gap.get("search_queries", [])
               state.refresh_needed = True
               return FlowStep.REFRESH
    
        return FlowStep.REASON

    async def _step_refresh(self, state: AgentState) -> FlowStep:
        await self.refresher.startup()
        state = await self.refresher.refresh(state)
        return FlowStep.RETRIEVE

    async def _step_reason(self, state: AgentState) -> FlowStep:
        # لود ایجنت استدلال‌گر (که احتمالا از Granite یا مدل دیگری استفاده می‌کند)
        # فرض بر این است که ReasonerAgent هم از الگوی lazy پیروی می‌کند
        
    
        async with self.config["llm_client_factory"]() as reasoning_model:
             # اگر متد reason در ایجنت شما مدل می‌گیرد، آن را پاس دهید
             # یا ایجنت را داخل بلاک بسازید
             reasoner = ReasonerAgent(self.config)#, model=reasoning_model)
             state = await reasoner.reason(state)
        
        return FlowStep.VALIDATE

    async def _step_validate(self, state: AgentState) -> FlowStep:
        result = await self.validator.validate(state)
        if state.iteration < state.max_iterations:
            if result["score"] >= 0.75:
                state.final_answer = result["answer"]
                state.confidence   = result["score"]
                return FlowStep.ANSWER
            elif result["score"] >= 0.50:
                return FlowStep.RETRIEVE
            else:
                return FlowStep.REFRESH
        else:    
            state.final_answer = result["answer"]
            state.confidence   = result["score"]
            return FlowStep.ANSWER

             
    def _transition(self, state: AgentState, next_step: FlowStep):
        allowed = self._transitions.get(state.current_step, [])
        if next_step not in allowed and next_step != FlowStep.ERROR:
            logger.error(f"Invalid transition {state.current_step} -> {next_step}")
            state.current_step = FlowStep.ERROR
            return
        state.current_step = next_step




# import time
# import logging
# from agents.state import AgentState, FlowStep
# from agents.retriever_agent import RetrieverAgent
# from agents.reasoner_agent import ReasonerAgent
# from agents.validator_agent import ValidatorAgent
# from agents.knowledge_refresh_agent import KnowledgeRefreshAgent
# from llm.prompt_understanding import Phi4MiniClient
# from agents.tool_registry import register_tool


# logger = logging.getLogger(__name__)
# class Orchestrator:
#     """
#     State Machine دستی - بدون هیچ framework خارجی
#     هر step به صورت صریح تعریف و کنترل می‌شود
#     """

#     def __init__(self, config: dict=[]):
#         self.config = config
#         self.phi4   = Phi4MiniClient(config)          # فهم + tool-call
#         self.retriever = RetrieverAgent(config)
#         self.reasoner  = ReasonerAgent(config)
#         self.validator = ValidatorAgent(config)
#         self.refresher = KnowledgeRefreshAgent(config)

#         # ثبت انتقال‌های مجاز
#         self._transitions: dict[FlowStep, list[FlowStep]] = {
#             FlowStep.START:      [FlowStep.UNDERSTAND],
#             FlowStep.UNDERSTAND: [FlowStep.RETRIEVE, FlowStep.ERROR],
#             FlowStep.RETRIEVE:   [FlowStep.ASSESS],
#             FlowStep.ASSESS:     [FlowStep.REASON, FlowStep.REFRESH],
#             FlowStep.REFRESH:    [FlowStep.RETRIEVE],
#             FlowStep.REASON:     [FlowStep.VALIDATE],
#             FlowStep.VALIDATE:   [FlowStep.ANSWER, FlowStep.RETRIEVE,
#                                   FlowStep.REFRESH],
#             FlowStep.ANSWER:     [],
#             FlowStep.ERROR:      [],
#         }

#     # در orchestrator یا یه فایل جداگانه
#     @register_tool("retrieve")
#     async def tool_retrieve(self,state: AgentState, query: str = "", **_) -> None:
#         q = query or state.query
#         state.retrieval = await self.retriever.retrieve(state, q)
    
#     @register_tool("detect_gaps")
#     async def tool_detect_gaps(self,state: AgentState, **_) -> None:
#         gaps = await self.phi4.detect_gaps(state.query, state.retrieval.context)
#         state.gaps = gaps
#         if not gaps["sufficient"]:
#             state.refresh_needed = True
    
#     @register_tool("reason")
#     async def tool_reason(self,state: AgentState, **_) -> None:
#         state.answer = await self.phi4.reason(state.query, state.retrieval.context)
    
#     @register_tool("sub_question")
#     async def tool_sub_question(self,state: AgentState, description: str = "", **_) -> None:
#         result = await self.retriever.retrieve(state, description)
#         state.retrieval.merge(result)
    
    
#     async def run(self, query: str, session_id: str) -> AgentState:
#         state = AgentState(query=query, session_id=session_id)
#         state.current_step = FlowStep.START

#         while state.current_step not in (FlowStep.ANSWER, FlowStep.ERROR):

#             if state.iteration > state.max_iterations:
#                 logger.warning("Max iterations reached")
#                 state.current_step = FlowStep.REASON
                

#             next_step = await self._execute_step(state)
#             self._transition(state, next_step)

#         return state

#     async def _execute_step(self, state: AgentState) -> FlowStep:
#         step = state.current_step
#         t0 = time.perf_counter()
#         logger.info(f"[{state.session_id}] Executing step: {step}")

#         try:
#             if step == FlowStep.START:
#                 next_step = FlowStep.UNDERSTAND

#             elif step == FlowStep.UNDERSTAND:
#                 next_step = await self._step_understand(state)

#             elif step == FlowStep.RETRIEVE:
#                 next_step = await self._step_retrieve(state)

#             elif step == FlowStep.ASSESS:
#                 next_step = await self._step_assess(state)

#             elif step == FlowStep.REFRESH:
#                 next_step = await self._step_refresh(state)

#             elif step == FlowStep.REASON:
#                 next_step = await self._step_reason(state)

#             elif step == FlowStep.VALIDATE:
#                 next_step = await self._step_validate(state)
                
#             elif step == FlowStep.ANSWER:
#                 next_step = await self._step_reason(state)
#             else:
#                 next_step = FlowStep.ERROR

#         except Exception as e:
#             state.errors.append(f"[{step}] {str(e)}")
#             logger.error(f"Step {step} failed: {e}", exc_info=True)
#             next_step = FlowStep.ERROR

#         state.timings[step.value] = time.perf_counter() - t0
#         state.iteration += 1
#         return next_step

#     # ─────────────────────────────────────────────
#     # STEP HANDLERS
#     # ─────────────────────────────────────────────

#     async def _step_understand(self, state: AgentState) -> FlowStep:
#         """
#         phi4-mini single call:
#         - intent detection
#         - tool-call generation
#         - sub-question decomposition
#         - execution plan
#         """
#         result = await self.phi4.understand_and_plan(
#             query=state.query,
#             history=state.history,
#         )
    
#         # result is UnderstandPlanResult dataclass — use dot access
#         state.intent             = result["intent"]
#         state.tool_calls         = result["tool_calls"]
#         state.sub_questions      = result["sub_questions"]
#         state.extracted_entities = result["entities"]
#         state.plan_steps         = result["steps"]        # execution plan هم اینجاست
#         state.strategy           = result["strategy"]
    
#         return FlowStep.RETRIEVE


#     async def _step_retrieve(self, state: AgentState) -> FlowStep:
#         await self.retriever.startup()
#         state = await self.retriever.retrieve(state)
#         return FlowStep.ASSESS

#     async def _step_assess(self, state: AgentState) -> FlowStep:
#         """
#         تصمیم ساده بر اساس confidence:
#         کافی ──► REASON
#         ناکافی ──► REFRESH
#         """
#         # c = state.retrieval.confidence
#         # if c >= self.config.get("confidence_threshold", 0.65):
#             # return FlowStep.REASON
#         # state.refresh_needed = True
#         # return FlowStep.REFRESH
    
        
#             # confidence-based check (موجود)
#         c = state.retrieval.confidence
#         if c < self.config.get("confidence_threshold", 0.65):
#             state.refresh_needed = True
#             return FlowStep.REFRESH
    
#         # gap detection با phi4 (جدید)
#         fused = "\n".join(
#             ch.get("content", "") for ch in state.retrieval.vector_chunks[:5]
#         )
#         gap = await self.phi4.detect_gaps(state.query, fused)
    
#         if not gap["sufficient"] and gap["confidence"] < 0.7:
#             state.refresh_queries = gap.get("search_queries", [])
#             state.refresh_needed = True
#             return FlowStep.REFRESH
    
#         return FlowStep.REASON
    
    

#     async def _step_refresh(self, state: AgentState) -> FlowStep:
#         await self.refresher.startup()
#         state = await self.refresher.refresh(state)
#         return FlowStep.RETRIEVE   # بعد از refresh دوباره retrieve

#     async def _step_reason(self, state: AgentState) -> FlowStep:
#         await self.reasoner.startup()
#         state = await self.reasoner.reason(state)
#         await self.reasoner.shutdown()
#         return FlowStep.VALIDATE

#     async def _step_validate(self, state: AgentState) -> FlowStep:
#         result = await self.validator.validate(state)
#         if result["score"] >= 0.75:
#             state.final_answer = result["answer"]
#             state.confidence   = result["score"]
#             return FlowStep.ANSWER
#         elif result["score"] >= 0.50:
#             return FlowStep.RETRIEVE
#         else:
#             return FlowStep.REFRESH

#     def _transition(self, state: AgentState, next_step: FlowStep):
#         allowed = self._transitions.get(state.current_step, [])
#         if next_step not in allowed and next_step != FlowStep.ERROR:
#             logger.error(
#                 f"Invalid transition {state.current_step} -> {next_step}"
#             )
#             state.current_step = FlowStep.ERROR
#             return
#         state.current_step = next_step



#=======planner from llm -=====================
# tool_fn = TOOL_REGISTRY.get(step["action"])
# if not tool_fn:
#     # fallback به retrieve ساده
#     tool_fn = TOOL_REGISTRY["retrieve"]
# class Orchestrator:
#     async def run(self, query: str) -> str:
#         state = AgentState(query=query)

#         # مرحله ۱: understand
#         understanding = await self.phi4.understand(query, [])
#         state.intent        = understanding["intent"]
#         state.sub_questions = understanding["sub_questions"]
#         state.extracted_entities = understanding["entities"]

#         # مرحله ۲: plan (LLM تصمیم میگیره چی اجرا بشه)
#         plan = await self.phi4.plan(query, understanding)
#         state.plan = plan

#         # مرحله ۳: execute plan dynamically
#         await self._execute_plan(state, plan)

#         # مرحله ۴: respond
#         return state.answer or await self.phi4.respond(state)

#     async def _execute_plan(self, state: AgentState, plan: dict) -> None:
#         steps = plan.get("steps", [])
#         completed: set[int] = set()

#         # dependency-aware execution
#         max_iterations = len(steps) * 2  # جلوگیری از infinite loop
#         iteration = 0

#         while len(completed) < len(steps) and iteration < max_iterations:
#             iteration += 1
#             for step in steps:
#                 sid = step["id"]
#                 if sid in completed:
#                     continue
#                 deps = step.get("depends_on", [])
#                 if not all(d in completed for d in deps):
#                     continue  # منتظر dependency

#                 tool_fn = TOOL_REGISTRY.get(step["action"])
#                 if tool_fn:
#                     await tool_fn(state, **step)
#                 completed.add(sid)
