# کنترل داینامیک فلو با Prompt Understanding

## 🎯 مشکل فعلی

در حال حاضر، Orchestrator یک فلوی **ثابت** دارد:
```
START → UNDERSTAND → RETRIEVE → ASSESS → REFRESH/REASON → VALIDATE → ANSWER
```

**محدودیت‌ها**:
1. ❌ همه query ها از یک مسیر می‌گذرند
2. ❌ نمی‌تواند مراحل را skip کند
3. ❌ نمی‌تواند مراحل را تکرار کند (به جز loop کلی)
4. ❌ نمی‌تواند مراحل جدید اضافه کند
5. ❌ تصمیم‌گیری فقط در ASSESS است

## 💡 راه‌حل: Hybrid Orchestration

ترکیب **فلوی ثابت** (برای reliability) با **کنترل داینامیک** (برای flexibility)

```
┌─────────────────────────────────────────────────────────────┐
│                    PROMPT UNDERSTANDING                      │
│  ─────────────────────────────────────────────────────────  │
│  1. Query Analysis                                          │
│  2. Intent Classification                                   │
│  3. Complexity Assessment                                   │
│  4. Strategy Selection                                      │
│  5. Dynamic Plan Generation                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ADAPTIVE ORCHESTRATOR                           │
│  ─────────────────────────────────────────────────────────  │
│  • Base Flow (ثابت برای reliability)                       │
│  • Dynamic Routing (داینامیک برای flexibility)             │
│  • Step Skipping (skip مراحل غیرضروری)                    │
│  • Conditional Branching (شاخه‌های شرطی)                   │
│  • Loop Control (کنترل تکرار)                              │
└─────────────────────────────────────────────────────────────┘
```

## 🏗️ معماری پیشنهادی

### 1. Query Classifier

```python
from enum import Enum
from dataclasses import dataclass

class QueryType(Enum):
    """انواع query بر اساس پیچیدگی و نیاز"""
    SIMPLE_FACT = "simple_fact"           # سوال ساده واقعی
    COMPLEX_REASONING = "complex_reasoning"  # نیاز به استدلال پیچیده
    MULTI_HOP = "multi_hop"               # نیاز به چند مرحله جستجو
    TEMPORAL = "temporal"                 # سوال زمانی (نیاز به refresh)
    COMPARATIVE = "comparative"           # مقایسه‌ای
    AGGREGATION = "aggregation"           # جمع‌آوری از چند منبع
    CREATIVE = "creative"                 # خلاقانه (کم‌تر نیاز به retrieval)

class QueryComplexity(Enum):
    LOW = "low"       # 1-2 مرحله
    MEDIUM = "medium" # 3-4 مرحله
    HIGH = "high"     # 5+ مرحله

@dataclass
class QueryAnalysis:
    """نتیجه تحلیل query"""
    query_type: QueryType
    complexity: QueryComplexity
    requires_refresh: bool
    requires_graph: bool
    requires_reasoning: bool
    estimated_steps: int
    confidence: float
    suggested_strategy: str
```

### 2. Enhanced Prompt Understanding

```python
class EnhancedPromptUnderstanding:
    """
    سیستم درک سوال پیشرفته که:
    1. Query را تحلیل می‌کند
    2. Strategy مناسب را انتخاب می‌کند
    3. Plan داینامیک تولید می‌کند
    """
    
    def __init__(self, phi4_client):
        self.phi4 = phi4_client
        self.query_classifier = QueryClassifier()
        self.strategy_selector = StrategySelector()
    
    async def analyze_and_plan(
        self,
        query: str,
        history: list[dict],
    ) -> EnhancedUnderstandingResult:
        """
        تحلیل جامع query و تولید plan داینامیک
        """
        # 1. تحلیل query
        analysis = await self.query_classifier.classify(query)
        
        # 2. استخراج اطلاعات پایه (موجود)
        base_understanding = await self.phi4.understand_and_plan(
            query=query,
            history=history,
        )
        
        # 3. انتخاب strategy
        strategy = self.strategy_selector.select(
            query_type=analysis.query_type,
            complexity=analysis.complexity,
            entities=base_understanding["entities"],
        )
        
        # 4. تولید plan داینامیک
        dynamic_plan = self._generate_dynamic_plan(
            analysis=analysis,
            strategy=strategy,
            base_understanding=base_understanding,
        )
        
        return EnhancedUnderstandingResult(
            # اطلاعات پایه
            intent=base_understanding["intent"],
            entities=base_understanding["entities"],
            tool_calls=base_understanding["tool_calls"],
            sub_questions=base_understanding["sub_questions"],
            
            # اطلاعات پیشرفته
            query_analysis=analysis,
            strategy=strategy,
            dynamic_plan=dynamic_plan,
            
            # کنترل فلو
            skip_steps=dynamic_plan.skip_steps,
            required_steps=dynamic_plan.required_steps,
            conditional_branches=dynamic_plan.conditional_branches,
        )
    
    def _generate_dynamic_plan(
        self,
        analysis: QueryAnalysis,
        strategy: Strategy,
        base_understanding: dict,
    ) -> DynamicPlan:
        """تولید plan بر اساس نوع query"""
        
        plan = DynamicPlan()
        
        # بر اساس query_type تصمیم می‌گیریم
        if analysis.query_type == QueryType.SIMPLE_FACT:
            # سوال ساده: skip reasoning پیچیده
            plan.skip_steps = [FlowStep.ASSESS, FlowStep.REFRESH]
            plan.required_steps = [FlowStep.RETRIEVE, FlowStep.REASON]
            plan.max_iterations = 1
            
        elif analysis.query_type == QueryType.TEMPORAL:
            # سوال زمانی: حتما refresh
            plan.required_steps = [FlowStep.REFRESH, FlowStep.RETRIEVE]
            plan.skip_steps = []
            plan.force_refresh = True
            
        elif analysis.query_type == QueryType.MULTI_HOP:
            # چند مرحله‌ای: تکرار retrieve
            plan.required_steps = [FlowStep.RETRIEVE, FlowStep.REASON]
            plan.max_retrieve_iterations = 3
            plan.enable_sub_questions = True
            
        elif analysis.query_type == QueryType.COMPLEX_REASONING:
            # استدلال پیچیده: همه مراحل
            plan.required_steps = list(FlowStep)
            plan.skip_steps = []
            plan.enable_chain_of_thought = True
            plan.enable_self_consistency = True
            
        elif analysis.query_type == QueryType.CREATIVE:
            # خلاقانه: کم‌تر retrieval
            plan.skip_steps = [FlowStep.REFRESH, FlowStep.ASSESS]
            plan.required_steps = [FlowStep.REASON]
            plan.retrieval_weight = 0.3  # وزن کم برای retrieval
            
        # شرط‌های اضافی
        if not analysis.requires_graph:
            plan.disable_graph_traversal = True
        
        if not analysis.requires_reasoning:
            plan.use_simple_answer = True
        
        return plan
```

### 3. Adaptive Orchestrator

```python
class AdaptiveOrchestrator(Orchestrator):
    """
    Orchestrator با قابلیت کنترل داینامیک فلو
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Enhanced understanding
        self.enhanced_understanding = EnhancedPromptUnderstanding(
            phi4_client=config["phi4_mini_factory"]
        )
        
        # Strategy executor
        self.strategy_executor = StrategyExecutor()
    
    async def run(self, query: str, session_id: str) -> AgentState:
        """
        اجرای adaptive با کنترل داینامیک
        """
        state = AgentState(query=query, session_id=session_id)
        state.current_step = FlowStep.START
        
        # مرحله UNDERSTAND با تحلیل پیشرفته
        understanding = await self.enhanced_understanding.analyze_and_plan(
            query=query,
            history=state.history,
        )
        
        # ذخیره در state
        self._populate_state_from_understanding(state, understanding)
        
        # اجرای plan داینامیک
        if understanding.dynamic_plan:
            return await self._execute_dynamic_plan(state, understanding)
        else:
            # fallback به فلوی ثابت
            return await super().run(query, session_id)
    
    async def _execute_dynamic_plan(
        self,
        state: AgentState,
        understanding: EnhancedUnderstandingResult,
    ) -> AgentState:
        """
        اجرای plan داینامیک
        """
        plan = understanding.dynamic_plan
        
        # مراحل اجباری
        for step in plan.required_steps:
            if self._should_execute_step(step, state, plan):
                next_step = await self._execute_step(state)
                
                # بررسی شرط‌های خروج زودهنگام
                if self._should_early_exit(state, plan):
                    break
                
                self._transition(state, next_step)
        
        # اگر به پاسخ نرسیدیم، از فلوی ثابت استفاده کن
        if state.current_step not in (FlowStep.ANSWER, FlowStep.ERROR):
            return await super().run(state.query, state.session_id)
        
        return state
    
    def _should_execute_step(
        self,
        step: FlowStep,
        state: AgentState,
        plan: DynamicPlan,
    ) -> bool:
        """
        آیا این step باید اجرا شود؟
        """
        # اگر در skip_steps باشد
        if step in plan.skip_steps:
            return False
        
        # شرط‌های خاص
        if step == FlowStep.REFRESH and not plan.force_refresh:
            # فقط اگر واقعا نیاز باشد
            return state.refresh_needed
        
        if step == FlowStep.ASSESS and plan.skip_assessment:
            return False
        
        return True
    
    def _should_early_exit(
        self,
        state: AgentState,
        plan: DynamicPlan,
    ) -> bool:
        """
        آیا می‌توانیم زودتر خارج شویم؟
        """
        # برای query های ساده
        if plan.allow_early_exit:
            if state.confidence >= 0.9:
                return True
        
        # اگر به جواب رسیدیم
        if state.final_answer and state.confidence >= 0.75:
            return True
        
        return False
```

### 4. Strategy Patterns

```python
class Strategy(Enum):
    """استراتژی‌های مختلف برای انواع query"""
    FAST_RETRIEVAL = "fast_retrieval"      # سریع، کم‌عمق
    DEEP_REASONING = "deep_reasoning"      # عمیق، استدلال پیچیده
    ITERATIVE_REFINEMENT = "iterative"     # تکراری، بهبود تدریجی
    GRAPH_FOCUSED = "graph_focused"        # تمرکز روی گراف
    FRESH_DATA = "fresh_data"              # داده تازه (refresh)
    HYBRID = "hybrid"                      # ترکیبی

@dataclass
class DynamicPlan:
    """Plan داینامیک برای اجرا"""
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
    enable_chain_of_thought: bool = True
    enable_self_consistency: bool = False
    
    # تنظیمات
    retrieval_weight: float = 0.7
    disable_graph_traversal: bool = False
    use_simple_answer: bool = False
    
    # شاخه‌های شرطی
    conditional_branches: dict = field(default_factory=dict)
```

## 📊 مثال‌های کاربردی

### مثال 1: سوال ساده واقعی

**Query**: "پایتخت فرانسه کجاست؟"

**تحلیل**:
```python
QueryAnalysis(
    query_type=QueryType.SIMPLE_FACT,
    complexity=QueryComplexity.LOW,
    requires_refresh=False,
    requires_graph=False,
    requires_reasoning=False,
    estimated_steps=2,
)
```

**Plan داینامیک**:
```python
DynamicPlan(
    required_steps=[FlowStep.RETRIEVE, FlowStep.REASON],
    skip_steps=[FlowStep.ASSESS, FlowStep.REFRESH],
    max_iterations=1,
    allow_early_exit=True,
)
```

**فلوی اجرا**:
```
UNDERSTAND → RETRIEVE → REASON → ANSWER
(skip: ASSESS, REFRESH, VALIDATE)
```

### مثال 2: سوال زمانی

**Query**: "آخرین قیمت سهام اپل چقدر است؟"

**تحلیل**:
```python
QueryAnalysis(
    query_type=QueryType.TEMPORAL,
    complexity=QueryComplexity.MEDIUM,
    requires_refresh=True,
    requires_graph=True,
    requires_reasoning=False,
    estimated_steps=4,
)
```

**Plan داینامیک**:
```python
DynamicPlan(
    required_steps=[FlowStep.REFRESH, FlowStep.RETRIEVE, FlowStep.REASON],
    skip_steps=[],
    force_refresh=True,
    max_iterations=1,
)
```

**فلوی اجرا**:
```
UNDERSTAND → REFRESH → RETRIEVE → REASON → VALIDATE → ANSWER
```

### مثال 3: سوال پیچیده چند مرحله‌ای

**Query**: "تفاوت بین سیاست‌های اقتصادی ترامپ و بایدن چیست و کدام موثرتر بود؟"

**تحلیل**:
```python
QueryAnalysis(
    query_type=QueryType.COMPLEX_REASONING,
    complexity=QueryComplexity.HIGH,
    requires_refresh=False,
    requires_graph=True,
    requires_reasoning=True,
    estimated_steps=6,
)
```

**Plan داینامیک**:
```python
DynamicPlan(
    required_steps=list(FlowStep),  # همه مراحل
    skip_steps=[],
    max_iterations=3,
    enable_sub_questions=True,
    enable_chain_of_thought=True,
    enable_self_consistency=True,
)
```

**فلوی اجرا**:
```
UNDERSTAND → 
  RETRIEVE (Trump policies) →
  RETRIEVE (Biden policies) →
  ASSESS → REASON (compare) →
  VALIDATE → 
  (if score < 0.75) RETRIEVE (more data) →
  REASON (refine) → VALIDATE →
ANSWER
```

### مثال 4: سوال خلاقانه

**Query**: "یک داستان کوتاه درباره یک ربات که عاشق می‌شود بنویس"

**تحلیل**:
```python
QueryAnalysis(
    query_type=QueryType.CREATIVE,
    complexity=QueryComplexity.LOW,
    requires_refresh=False,
    requires_graph=False,
    requires_reasoning=True,
    estimated_steps=1,
)
```

**Plan داینامیک**:
```python
DynamicPlan(
    required_steps=[FlowStep.REASON],
    skip_steps=[FlowStep.RETRIEVE, FlowStep.ASSESS, FlowStep.REFRESH],
    max_iterations=1,
    use_simple_answer=True,
    retrieval_weight=0.1,  # خیلی کم
)
```

**فلوی اجرا**:
```
UNDERSTAND → REASON → ANSWER
(skip: RETRIEVE, ASSESS, REFRESH, VALIDATE)
```

## 🔄 ترکیب با Knowledge Refresh

```python
class AdaptiveKnowledgeRefresh:
    """
    Knowledge refresh هوشمند بر اساس query type
    """
    
    async def should_refresh(
        self,
        state: AgentState,
        query_analysis: QueryAnalysis,
    ) -> RefreshDecision:
        """
        تصمیم هوشمند برای refresh
        """
        # سوالات زمانی: حتما refresh
        if query_analysis.query_type == QueryType.TEMPORAL:
            return RefreshDecision(
                should_refresh=True,
                reason="temporal_query",
                priority="high",
                domains=[DataDomain.NEWS, DataDomain.FINANCIAL],
            )
        
        # confidence پایین: refresh
        if state.retrieval.confidence < 0.5:
            return RefreshDecision(
                should_refresh=True,
                reason="low_confidence",
                priority="medium",
                domains=self._infer_domains(state),
            )
        
        # gap detection: refresh هدفمند
        if state.knowledge_gaps:
            return RefreshDecision(
                should_refresh=True,
                reason="knowledge_gaps",
                priority="high",
                specific_queries=state.knowledge_gaps,
            )
        
        # default: no refresh
        return RefreshDecision(should_refresh=False)
    
    async def execute_targeted_refresh(
        self,
        decision: RefreshDecision,
        state: AgentState,
    ):
        """
        Refresh هدفمند فقط برای domain های مورد نیاز
        """
        if decision.specific_queries:
            # Refresh با query های خاص
            for query in decision.specific_queries:
                await self.refresh_agent.incremental_refresh(
                    state=state,
                    query=query,
                    domains=decision.domains,
                )
        else:
            # Refresh عمومی
            await self.refresh_agent.refresh(state)
```

## 📈 مزایای سیستم Adaptive

### 1. کارایی بهتر
- ⚡ سوالات ساده: 50-70% سریعتر (skip مراحل غیرضروری)
- 🎯 سوالات پیچیده: دقت بالاتر (مراحل بیشتر)
- 💾 مصرف منابع: بهینه (فقط آنچه لازم است)

### 2. انعطاف‌پذیری
- 🔀 مسیرهای مختلف برای query های مختلف
- 🔄 قابلیت تکرار و بازگشت
- ➕ قابلیت افزودن strategy های جدید

### 3. کنترل بهتر
- 🎛️ کنترل دقیق روی هر مرحله
- 📊 متریک‌های دقیق‌تر
- 🐛 Debug آسان‌تر

## 🚀 پیاده‌سازی تدریجی

### فاز 1: Query Classification (1 هفته)
```python
- [ ] QueryClassifier
- [ ] QueryAnalysis dataclass
- [ ] Integration با Phi4
```

### فاز 2: Strategy Selection (1 هفته)
```python
- [ ] StrategySelector
- [ ] Strategy patterns
- [ ] DynamicPlan generation
```

### فاز 3: Adaptive Orchestrator (2 هفته)
```python
- [ ] AdaptiveOrchestrator
- [ ] Dynamic plan execution
- [ ] Early exit logic
- [ ] Conditional branching
```

### فاز 4: Integration & Testing (1 هفته)
```python
- [ ] Integration با ImprovedKnowledgeRefreshAgent
- [ ] End-to-end testing
- [ ] Performance benchmarking
- [ ] Documentation
```

## 💡 نتیجه‌گیری

با این بهبود:
- ✅ **Reliability**: فلوی ثابت به عنوان fallback
- ✅ **Flexibility**: کنترل داینامیک برای بهینه‌سازی
- ✅ **Intelligence**: تصمیم‌گیری هوشمند بر اساس query
- ✅ **Efficiency**: skip مراحل غیرضروری
- ✅ **Scalability**: افزودن strategy های جدید آسان

سیستم می‌تواند به طور هوشمند تصمیم بگیرد که:
- کدام مراحل اجرا شوند
- چند بار تکرار شوند
- چه زمانی زودتر خارج شود
- چه strategy ای استفاده شود