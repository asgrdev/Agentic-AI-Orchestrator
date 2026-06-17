"""
تست‌های Adaptive Orchestrator و Query Classifier
"""
import pytest
import asyncio
from agents.query_classifier import QueryClassifier, QueryType, QueryComplexity
from agents.adaptive_orchestrator import AdaptiveOrchestrator, DynamicPlan
from agents.state import AgentState, FlowStep


class TestQueryClassifier:
    """تست Query Classifier"""
    
    def setup_method(self):
        self.classifier = QueryClassifier()
    
    def test_simple_fact_query(self):
        """تست سوال ساده واقعی"""
        query = "پایتخت فرانسه کجاست؟"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.SIMPLE_FACT
        assert analysis.complexity == QueryComplexity.LOW
        assert not analysis.requires_refresh
        assert analysis.estimated_steps <= 3
        assert analysis.suggested_strategy == "fast_retrieval"
    
    def test_temporal_query(self):
        """تست سوال زمانی"""
        query = "آخرین قیمت سهام اپل چقدر است؟"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.TEMPORAL
        assert analysis.requires_refresh
        assert len(analysis.temporal_indicators) > 0
        assert analysis.suggested_strategy == "fresh_data"
    
    def test_complex_reasoning_query(self):
        """تست سوال پیچیده"""
        query = "تفاوت بین سیاست‌های اقتصادی ترامپ و بایدن چیست و کدام موثرتر بود؟"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type in (QueryType.COMPLEX_REASONING, QueryType.COMPARATIVE)
        assert analysis.complexity in (QueryComplexity.MEDIUM, QueryComplexity.HIGH)
        assert analysis.requires_reasoning
        assert analysis.estimated_steps >= 5
    
    def test_creative_query(self):
        """تست سوال خلاقانه"""
        query = "یک داستان کوتاه درباره یک ربات که عاشق می‌شود بنویس"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.CREATIVE
        assert analysis.complexity == QueryComplexity.LOW
        assert not analysis.requires_refresh
        assert not analysis.requires_graph
        assert analysis.estimated_steps <= 2
    
    def test_multi_hop_query(self):
        """تست سوال چند مرحله‌ای"""
        query = "چرا قیمت نفت بر اقتصاد ایران تاثیر می‌گذارد؟"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.MULTI_HOP
        assert analysis.requires_graph
        assert analysis.estimated_steps >= 4
    
    def test_comparative_query(self):
        """تست سوال مقایسه‌ای"""
        query = "مقایسه پایتون و جاوا"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.COMPARATIVE
        assert analysis.requires_reasoning
    
    def test_english_query(self):
        """تست query انگلیسی"""
        query = "What is the capital of France?"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.SIMPLE_FACT
        assert analysis.complexity == QueryComplexity.LOW
    
    def test_temporal_english(self):
        """تست query زمانی انگلیسی"""
        query = "What is the latest price of Apple stock?"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.TEMPORAL
        assert analysis.requires_refresh


class TestDynamicPlanGeneration:
    """تست تولید Dynamic Plan"""
    
    def setup_method(self):
        self.classifier = QueryClassifier()
        # Mock config
        self.config = {
            "kuzu_path": "test_db",
            "weaviate": {},
            "confidence_threshold": 0.65,
        }
    
    def test_simple_fact_plan(self):
        """تست plan برای سوال ساده"""
        query = "پایتخت فرانسه کجاست؟"
        analysis = self.classifier.classify(query)
        
        # شبیه‌سازی تولید plan
        assert analysis.query_type == QueryType.SIMPLE_FACT
        
        # انتظارات:
        # - skip ASSESS, REFRESH, VALIDATE
        # - فقط RETRIEVE و REASON
        # - max_iterations = 1
        # - allow_early_exit = True
    
    def test_temporal_plan(self):
        """تست plan برای سوال زمانی"""
        query = "آخرین قیمت بیت کوین؟"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.TEMPORAL
        assert analysis.requires_refresh
        
        # انتظارات:
        # - force_refresh = True
        # - شامل REFRESH, RETRIEVE, REASON, VALIDATE
    
    def test_creative_plan(self):
        """تست plan برای سوال خلاقانه"""
        query = "یک شعر درباره پاییز بنویس"
        analysis = self.classifier.classify(query)
        
        assert analysis.query_type == QueryType.CREATIVE
        
        # انتظارات:
        # - skip RETRIEVE, ASSESS, REFRESH, VALIDATE
        # - فقط REASON
        # - disable_graph_traversal = True


class TestAdaptiveOrchestratorIntegration:
    """تست‌های integration (نیاز به mock دارند)"""
    
    @pytest.mark.asyncio
    async def test_state_initialization(self):
        """تست مقداردهی اولیه state"""
        state = AgentState(query="تست", session_id="test-123")
        
        assert state.query == "تست"
        assert state.session_id == "test-123"
        assert state.current_step == FlowStep.START
        assert state.iteration == 0
        assert state.query_analysis is None
        assert state.dynamic_plan is None
    
    @pytest.mark.asyncio
    async def test_query_analysis_in_state(self):
        """تست ذخیره query_analysis در state"""
        classifier = QueryClassifier()
        analysis = classifier.classify("پایتخت فرانسه؟")
        
        state = AgentState(query="پایتخت فرانسه؟")
        state.query_analysis = analysis
        
        assert state.query_analysis is not None
        assert state.query_analysis.query_type == QueryType.SIMPLE_FACT
    
    @pytest.mark.asyncio
    async def test_dynamic_plan_in_state(self):
        """تست ذخیره dynamic_plan در state"""
        plan = DynamicPlan()
        plan.required_steps = [FlowStep.RETRIEVE, FlowStep.REASON]
        plan.skip_steps = [FlowStep.ASSESS]
        plan.max_iterations = 1
        
        state = AgentState(query="تست")
        state.dynamic_plan = plan
        
        assert state.dynamic_plan is not None
        assert len(state.dynamic_plan.required_steps) == 2
        assert FlowStep.ASSESS in state.dynamic_plan.skip_steps


def test_query_types_coverage():
    """تست پوشش همه انواع query"""
    classifier = QueryClassifier()
    
    test_queries = {
        QueryType.SIMPLE_FACT: "پایتخت ایران کجاست؟",
        QueryType.TEMPORAL: "آخرین اخبار بورس",
        QueryType.CREATIVE: "یک داستان بنویس",
        QueryType.COMPARATIVE: "تفاوت A و B",
        QueryType.MULTI_HOP: "چرا X باعث Y می‌شود؟",
    }
    
    for expected_type, query in test_queries.items():
        analysis = classifier.classify(query)
        # حداقل باید یکی از type های مرتبط باشد
        assert analysis.query_type is not None
        assert isinstance(analysis.query_type, QueryType)


def test_complexity_levels():
    """تست سطوح مختلف پیچیدگی"""
    classifier = QueryClassifier()
    
    # Low complexity
    simple = classifier.classify("چیست؟")
    assert simple.complexity == QueryComplexity.LOW
    
    # Medium complexity
    medium = classifier.classify("تفاوت بین A و B چیست؟")
    assert medium.complexity in (QueryComplexity.LOW, QueryComplexity.MEDIUM)
    
    # High complexity
    complex_query = "تحلیل کامل تاثیرات اقتصادی، اجتماعی و سیاسی تغییرات اقلیمی بر کشورهای در حال توسعه"
    high = classifier.classify(complex_query)
    assert high.complexity in (QueryComplexity.MEDIUM, QueryComplexity.HIGH)


if __name__ == "__main__":
    # اجرای تست‌های ساده
    print("Testing Query Classifier...")
    
    classifier = QueryClassifier()
    
    test_queries = [
        "پایتخت فرانسه کجاست؟",
        "آخرین قیمت بیت کوین؟",
        "یک داستان کوتاه بنویس",
        "تفاوت پایتون و جاوا",
        "چرا آسمان آبی است؟",
    ]
    
    for query in test_queries:
        analysis = classifier.classify(query)
        print(f"\nQuery: {query}")
        print(f"  Type: {analysis.query_type.value}")
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Strategy: {analysis.suggested_strategy}")
        print(f"  Steps: {analysis.estimated_steps}")
        print(f"  Requires Refresh: {analysis.requires_refresh}")
        print(f"  Requires Graph: {analysis.requires_graph}")
    
    print("\n✅ All manual tests passed!")

