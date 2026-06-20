"""
Query Classifier - طبقه‌بندی و تحلیل query

این ماژول query را تحلیل کرده و:
1. نوع query را تشخیص می‌دهد
2. پیچیدگی را ارزیابی می‌کند
3. نیازمندی‌ها را مشخص می‌کند
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """انواع query بر اساس پیچیدگی و نیاز"""
    SIMPLE_FACT = "simple_fact"              # سوال ساده واقعی
    COMPLEX_REASONING = "complex_reasoning"  # نیاز به استدلال پیچیده
    MULTI_HOP = "multi_hop"                  # نیاز به چند مرحله جستجو
    TEMPORAL = "temporal"                    # سوال زمانی (نیاز به refresh)
    COMPARATIVE = "comparative"              # مقایسه‌ای
    AGGREGATION = "aggregation"              # جمع‌آوری از چند منبع
    CREATIVE = "creative"                    # خلاقانه (کم‌تر نیاز به retrieval)
    CALCULATION = "calculation"              # محاسبات ریاضی


class QueryComplexity(Enum):
    """سطح پیچیدگی query"""
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
    keywords: list[str] = field(default_factory=list)
    temporal_indicators: list[str] = field(default_factory=list)


class QueryClassifier:
    """
    طبقه‌بندی query بر اساس pattern matching و heuristics
    """
    
    # الگوهای زمانی
    TEMPORAL_PATTERNS = [
        r'\b(آخرین|جدیدترین|اخیر|امروز|دیروز|فردا|الان|اکنون)\b',
        r'\b(latest|recent|current|today|yesterday|now)\b',
        r'\b(قیمت|price|rate|نرخ)\b.*\b(فعلی|current|now)\b',
        r'\b\d{4}\b',  # سال
    ]
    
    # الگوهای مقایسه‌ای
    COMPARATIVE_PATTERNS = [
        r'\b(تفاوت|مقایسه|compare|difference|versus|vs)\b',
        r'\b(بهتر|بدتر|بیشتر|کمتر|better|worse|more|less)\b',
        r'\b(بین|between)\b.*\b(و|and)\b',
    ]
    
    # الگوهای خلاقانه
    CREATIVE_PATTERNS = [
        r'\b(بنویس|write|create|generate|تولید|ایجاد)\b',
        r'\b(داستان|story|شعر|poem|نامه|letter)\b',
        r'\b(تصور کن|imagine|فرض کن|suppose)\b',
    ]
    
    # الگوهای چند مرحله‌ای
    MULTI_HOP_PATTERNS = [
        r'\b(چرا|why|چگونه|how)\b',
        r'\b(تاثیر|effect|impact|نتیجه|result)\b',
        r'\b(رابطه|relation|ارتباط|connection)\b',
    ]
    
    # الگوهای محاسباتی
    CALCULATION_PATTERNS = [
        r'\d+\s*%\s*(of|از)\s*\d+',
        r'\d+\s*[\+\-\*\/\^]\s*\d+',
        r'\b(calculate|compute|محاسبه|حساب)\b',
        r'\b(sqrt|square root|جذر|ریشه)\b',
        r'\b(sum|average|mean|total|جمع|میانگین)\b.*\d',
        r'^\s*\d[\d\s\+\-\*\/\.\%\(\)]+\d\s*$',
    ]

    # کلمات کلیدی ساده
    SIMPLE_KEYWORDS = [
        'چیست', 'کیست', 'کجاست', 'what', 'who', 'where', 'when',
        'پایتخت', 'capital', 'نام', 'name',
    ]
    
    def __init__(self):
        self._temporal_re = re.compile('|'.join(self.TEMPORAL_PATTERNS), re.IGNORECASE)
        self._comparative_re = re.compile('|'.join(self.COMPARATIVE_PATTERNS), re.IGNORECASE)
        self._creative_re = re.compile('|'.join(self.CREATIVE_PATTERNS), re.IGNORECASE)
        self._multi_hop_re = re.compile('|'.join(self.MULTI_HOP_PATTERNS), re.IGNORECASE)
        self._calculation_re = re.compile('|'.join(self.CALCULATION_PATTERNS), re.IGNORECASE)
    
    def classify(self, query: str) -> QueryAnalysis:
        """
        طبقه‌بندی query
        
        Args:
            query: متن سوال
            
        Returns:
            QueryAnalysis با اطلاعات کامل
        """
        query_lower = query.lower()
        
        # استخراج keywords
        keywords = self._extract_keywords(query)
        
        # تشخیص temporal indicators
        temporal_indicators = self._temporal_re.findall(query)
        
        # تشخیص نوع query
        query_type = self._detect_query_type(query, query_lower, temporal_indicators)
        
        # محاسبه پیچیدگی
        complexity = self._assess_complexity(query, query_type)
        
        # تعیین نیازمندی‌ها
        requires_refresh = self._needs_refresh(query_type, temporal_indicators)
        requires_graph = self._needs_graph(query_type, complexity)
        requires_reasoning = self._needs_reasoning(query_type, complexity)
        
        # تخمین تعداد مراحل
        estimated_steps = self._estimate_steps(query_type, complexity)
        
        # پیشنهاد strategy
        suggested_strategy = self._suggest_strategy(query_type, complexity)
        
        # محاسبه confidence
        confidence = self._calculate_confidence(query_type, keywords)
        
        return QueryAnalysis(
            query_type=query_type,
            complexity=complexity,
            requires_refresh=requires_refresh,
            requires_graph=requires_graph,
            requires_reasoning=requires_reasoning,
            estimated_steps=estimated_steps,
            confidence=confidence,
            suggested_strategy=suggested_strategy,
            keywords=keywords,
            temporal_indicators=temporal_indicators,
        )
    
    def _detect_query_type(
        self,
        query: str,
        query_lower: str,
        temporal_indicators: list[str],
    ) -> QueryType:
        """تشخیص نوع query"""
        
        # 0. Calculation
        if self._calculation_re.search(query):
            return QueryType.CALCULATION

        # 1. Creative
        if self._creative_re.search(query):
            return QueryType.CREATIVE
        
        # 2. Temporal
        if temporal_indicators or self._temporal_re.search(query):
            return QueryType.TEMPORAL
        
        # 3. Comparative
        if self._comparative_re.search(query):
            return QueryType.COMPARATIVE
        
        # 4. Multi-hop
        if self._multi_hop_re.search(query):
            return QueryType.MULTI_HOP
        
        # 5. Simple fact
        if any(kw in query_lower for kw in self.SIMPLE_KEYWORDS):
            if len(query.split()) < 10:  # کوتاه
                return QueryType.SIMPLE_FACT
        
        # 6. Aggregation (چند entity یا چند سوال)
        if query.count('و') > 2 or query.count('and') > 2:
            return QueryType.AGGREGATION
        
        # 7. Default: Complex reasoning
        return QueryType.COMPLEX_REASONING
    
    def _assess_complexity(
        self,
        query: str,
        query_type: QueryType,
    ) -> QueryComplexity:
        """ارزیابی پیچیدگی"""
        
        # بر اساس نوع
        if query_type in (QueryType.SIMPLE_FACT, QueryType.CREATIVE, QueryType.CALCULATION):
            return QueryComplexity.LOW
        
        if query_type in (QueryType.TEMPORAL, QueryType.COMPARATIVE):
            return QueryComplexity.MEDIUM
        
        # بر اساس طول
        word_count = len(query.split())
        if word_count < 8:
            return QueryComplexity.LOW
        elif word_count < 15:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.HIGH
    
    def _needs_refresh(
        self,
        query_type: QueryType,
        temporal_indicators: list[str],
    ) -> bool:
        """آیا نیاز به refresh دارد؟"""
        if query_type == QueryType.TEMPORAL:
            return True
        if temporal_indicators:
            return True
        return False
    
    def _needs_graph(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
    ) -> bool:
        """آیا نیاز به graph traversal دارد؟"""
        if query_type in (QueryType.CREATIVE, QueryType.CALCULATION):
            return False
        if query_type in (QueryType.MULTI_HOP, QueryType.COMPARATIVE):
            return True
        if complexity == QueryComplexity.HIGH:
            return True
        return True  # default: استفاده از graph
    
    def _needs_reasoning(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
    ) -> bool:
        """آیا نیاز به reasoning پیچیده دارد؟"""
        if query_type in (QueryType.SIMPLE_FACT, QueryType.CALCULATION):
            return False
        return True
    
    def _estimate_steps(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
    ) -> int:
        """تخمین تعداد مراحل"""
        base_steps = {
            QueryType.SIMPLE_FACT: 2,
            QueryType.CREATIVE: 1,
            QueryType.CALCULATION: 1,
            QueryType.TEMPORAL: 4,
            QueryType.COMPARATIVE: 5,
            QueryType.MULTI_HOP: 6,
            QueryType.AGGREGATION: 5,
            QueryType.COMPLEX_REASONING: 6,
        }
        
        steps = base_steps.get(query_type, 4)
        
        # تعدیل بر اساس complexity
        if complexity == QueryComplexity.HIGH:
            steps += 2
        elif complexity == QueryComplexity.LOW:
            steps = max(1, steps - 1)
        
        return steps
    
    def _suggest_strategy(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
    ) -> str:
        """پیشنهاد strategy"""
        if query_type == QueryType.CALCULATION:
            return "direct_skill"
        elif query_type == QueryType.SIMPLE_FACT:
            return "fast_retrieval"
        elif query_type == QueryType.CREATIVE:
            return "direct_generation"
        elif query_type == QueryType.TEMPORAL:
            return "fresh_data"
        elif query_type == QueryType.MULTI_HOP:
            return "graph_focused"
        elif complexity == QueryComplexity.HIGH:
            return "deep_reasoning"
        else:
            return "hybrid"
    
    def _extract_keywords(self, query: str) -> list[str]:
        """استخراج کلمات کلیدی"""
        # حذف stop words ساده
        stop_words = {
            'است', 'بود', 'که', 'را', 'به', 'از', 'در', 'با',
            'is', 'was', 'the', 'a', 'an', 'to', 'of', 'in', 'for',
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # حداکثر 10 keyword
    
    def _calculate_confidence(
        self,
        query_type: QueryType,
        keywords: list[str],
    ) -> float:
        """محاسبه confidence در classification"""
        # confidence بالا برای pattern های واضح
        if query_type in (QueryType.CREATIVE, QueryType.TEMPORAL, QueryType.CALCULATION):
            return 0.9
        
        # confidence متوسط برای بقیه
        if len(keywords) >= 3:
            return 0.75
        
        return 0.6

