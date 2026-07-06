from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from datetime import datetime


class FlowStep(str, Enum):
    START            = "start"
    UNDERSTAND       = "understand"       # phi4-mini
    RETRIEVE         = "retrieve"         # Neo4j + Weaviate + OpenSearch
    ASSESS           = "assess"           # آیا دانش کافیه؟
    REFRESH          = "refresh"          # رفرش از منابع خارجی
    REASON           = "reason"           # granite4-7b
    VALIDATE         = "validate"
    ANSWER           = "answer"
    ERROR            = "error"


@dataclass
class RetrievalContext:
    graph_nodes:     list[dict] = field(default_factory=list)
    graph_edges:     list[dict] = field(default_factory=list)
    vector_chunks:   list[dict] = field(default_factory=list)
    bm25_docs:       list[dict] = field(default_factory=list)
    fused_context:   str = ""
    confidence:      float = 0.0


@dataclass
class AgentState:
    # ورودی
    query:               str = ""
    session_id:          str = ""
    history:             list[dict] = field(default_factory=list)

    # مرحله جاری
    current_step:        FlowStep = FlowStep.START
    iteration:           int = 0
    max_iterations:      int = 2

    # درک سوال
    intent:              str = ""
    extracted_entities:  list[dict] = field(default_factory=list)  # از NER
    linked_entities:     list[dict] = field(default_factory=list)  # Entity Linking
    tool_calls:          list[dict] = field(default_factory=list)  # از phi4-mini
    plan_steps:          list[dict] = field(default_factory=list)  # از phi4-mini
    strategy:            str = "sequential"                        # از phi4-mini
    sub_questions:       list[str] = field(default_factory=list)

    # بازیابی
    retrieval:           RetrievalContext = field(
                             default_factory=RetrievalContext
                         )
    knowledge_gaps:      list[str] = field(default_factory=list)
    refresh_needed:      bool = False
    refresh_attempts:    int = 0

    # خروجی
    reasoning_trace:     list[str] = field(default_factory=list)
    final_answer:        str = ""
    citations:           list[dict] = field(default_factory=list)
    confidence:          float = 0.0

    # متا
    errors:              list[str] = field(default_factory=list)
    timings:             dict[str, float] = field(default_factory=dict)
    # trace مرتبِ گره‌های طی‌شده برای نمایش گراف فلو (tab «Flow»)
    # هر ورودی: {step, duration, status, detail, ts}
    flow_trace:          list[dict] = field(default_factory=list)
    # «فرایند فکر» قابل‌نمایش در چت — بلاک‌های <think> مدل‌ها + خلاصه‌ی
    # تصمیم‌های orchestrator؛ هر ورودی: {stage, text, ts}
    thinking_log:        list[dict] = field(default_factory=list)
    metadata:            dict[str, Any] = field(default_factory=dict)
    created_at:          datetime = field(default_factory=datetime.utcnow)
    
    # Adaptive Orchestrator (جدید)
    query_analysis:      Any = None  # QueryAnalysis
    
    # Tool/Skill Execution (جدید)
    tool_results:        list[dict] = field(default_factory=list)  # نتایج اجرای tool calls
    step_results:        dict[int, dict] = field(default_factory=dict)  # نتایج اجرای plan steps
    
    # Validation (جدید)
    validation:          Any = None  # ValidationResult
    dynamic_plan:        Any = None  # DynamicPlan
