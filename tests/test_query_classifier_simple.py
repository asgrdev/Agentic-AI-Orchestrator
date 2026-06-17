"""
تست ساده Query Classifier (بدون pytest)
"""
import sys
sys.path.insert(0, '/Users/dbk/Desktop/agentic-graph-RAG')

from agents.query_classifier import QueryClassifier, QueryType, QueryComplexity


def test_simple_fact():
    """تست سوال ساده"""
    classifier = QueryClassifier()
    query = "پایتخت فرانسه کجاست؟"
    analysis = classifier.classify(query)
    
    print(f"✓ Query: {query}")
    print(f"  Type: {analysis.query_type.value}")
    print(f"  Complexity: {analysis.complexity.value}")
    print(f"  Strategy: {analysis.suggested_strategy}")
    print(f"  Steps: {analysis.estimated_steps}")
    
    assert analysis.query_type == QueryType.SIMPLE_FACT, f"Expected SIMPLE_FACT, got {analysis.query_type}"
    assert analysis.complexity == QueryComplexity.LOW, f"Expected LOW, got {analysis.complexity}"
    assert not analysis.requires_refresh, "Should not require refresh"
    print("  ✅ PASSED\n")


def test_temporal():
    """تست سوال زمانی"""
    classifier = QueryClassifier()
    query = "آخرین قیمت بیت کوین چقدر است؟"
    analysis = classifier.classify(query)
    
    print(f"✓ Query: {query}")
    print(f"  Type: {analysis.query_type.value}")
    print(f"  Complexity: {analysis.complexity.value}")
    print(f"  Requires Refresh: {analysis.requires_refresh}")
    print(f"  Temporal Indicators: {analysis.temporal_indicators}")
    
    assert analysis.query_type == QueryType.TEMPORAL, f"Expected TEMPORAL, got {analysis.query_type}"
    assert analysis.requires_refresh, "Should require refresh"
    assert len(analysis.temporal_indicators) > 0, "Should have temporal indicators"
    print("  ✅ PASSED\n")


def test_creative():
    """تست سوال خلاقانه"""
    classifier = QueryClassifier()
    query = "یک داستان کوتاه درباره یک ربات عاشق بنویس"
    analysis = classifier.classify(query)
    
    print(f"✓ Query: {query}")
    print(f"  Type: {analysis.query_type.value}")
    print(f"  Complexity: {analysis.complexity.value}")
    print(f"  Requires Graph: {analysis.requires_graph}")
    
    assert analysis.query_type == QueryType.CREATIVE, f"Expected CREATIVE, got {analysis.query_type}"
    assert not analysis.requires_graph, "Should not require graph"
    print("  ✅ PASSED\n")


def test_comparative():
    """تست سوال مقایسه‌ای"""
    classifier = QueryClassifier()
    query = "تفاوت بین پایتون و جاوا چیست؟"
    analysis = classifier.classify(query)
    
    print(f"✓ Query: {query}")
    print(f"  Type: {analysis.query_type.value}")
    print(f"  Complexity: {analysis.complexity.value}")
    print(f"  Requires Reasoning: {analysis.requires_reasoning}")
    
    assert analysis.query_type == QueryType.COMPARATIVE, f"Expected COMPARATIVE, got {analysis.query_type}"
    assert analysis.requires_reasoning, "Should require reasoning"
    print("  ✅ PASSED\n")


def test_multi_hop():
    """تست سوال چند مرحله‌ای"""
    classifier = QueryClassifier()
    query = "چرا افزایش قیمت نفت بر اقتصاد ایران تاثیر می‌گذارد؟"
    analysis = classifier.classify(query)
    
    print(f"✓ Query: {query}")
    print(f"  Type: {analysis.query_type.value}")
    print(f"  Complexity: {analysis.complexity.value}")
    print(f"  Estimated Steps: {analysis.estimated_steps}")
    
    assert analysis.query_type == QueryType.MULTI_HOP, f"Expected MULTI_HOP, got {analysis.query_type}"
    assert analysis.estimated_steps >= 4, f"Expected >= 4 steps, got {analysis.estimated_steps}"
    print("  ✅ PASSED\n")


def test_english_queries():
    """تست query های انگلیسی"""
    classifier = QueryClassifier()
    
    queries = [
        ("What is the capital of France?", QueryType.SIMPLE_FACT),
        ("What is the latest price of Apple stock?", QueryType.TEMPORAL),
        ("Write a short story about a robot", QueryType.CREATIVE),
        ("Compare Python and Java", QueryType.COMPARATIVE),
    ]
    
    for query, expected_type in queries:
        analysis = classifier.classify(query)
        print(f"✓ Query: {query}")
        print(f"  Type: {analysis.query_type.value}")
        assert analysis.query_type == expected_type, f"Expected {expected_type}, got {analysis.query_type}"
        print("  ✅ PASSED\n")


def test_complexity_levels():
    """تست سطوح پیچیدگی"""
    classifier = QueryClassifier()
    
    # Low
    simple = classifier.classify("چیست؟")
    print(f"✓ Simple query complexity: {simple.complexity.value}")
    assert simple.complexity == QueryComplexity.LOW
    
    # Medium
    medium = classifier.classify("تفاوت بین A و B چیست و کدام بهتر است؟")
    print(f"✓ Medium query complexity: {medium.complexity.value}")
    assert medium.complexity in (QueryComplexity.LOW, QueryComplexity.MEDIUM)
    
    # High
    complex_q = "تحلیل کامل تاثیرات اقتصادی، اجتماعی و سیاسی تغییرات اقلیمی بر کشورهای در حال توسعه در دهه آینده"
    high = classifier.classify(complex_q)
    print(f"✓ Complex query complexity: {high.complexity.value}")
    assert high.complexity in (QueryComplexity.MEDIUM, QueryComplexity.HIGH)
    
    print("  ✅ ALL COMPLEXITY TESTS PASSED\n")


def run_all_tests():
    """اجرای همه تست‌ها"""
    print("="*60)
    print("Testing Query Classifier")
    print("="*60 + "\n")
    
    try:
        test_simple_fact()
        test_temporal()
        test_creative()
        test_comparative()
        test_multi_hop()
        test_english_queries()
        test_complexity_levels()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

