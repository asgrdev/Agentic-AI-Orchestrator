#!/usr/bin/env python3
"""
Simple System Integration Test
Tests core components without heavy dependencies
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.query_classifier import QueryClassifier, QueryType
from agents.skill_executor import get_global_registry, SkillExecutor
from agents.state import AgentState


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


async def test_query_classifier():
    """Test Query Classifier"""
    print_section("TEST 1: QUERY CLASSIFIER")
    
    classifier = QueryClassifier()
    
    test_cases = [
        {
            "query": "What is machine learning?",
            "expected": QueryType.SIMPLE_FACT,
            "description": "Simple factual question"
        },
        {
            "query": "Compare supervised and unsupervised learning",
            "expected": QueryType.COMPARATIVE,
            "description": "Comparison query"
        },
        {
            "query": "How has AI evolved from 2010 to 2020?",
            "expected": QueryType.TEMPORAL,
            "description": "Temporal query"
        },
        {
            "query": "Write a creative story about robots",
            "expected": QueryType.CREATIVE,
            "description": "Creative task"
        },
        {
            "query": "What is the relationship between AI, ML, and DL?",
            "expected": QueryType.MULTI_HOP,
            "description": "Multi-hop reasoning"
        },
        {
            "query": "Calculate the average of 10, 20, 30",
            "expected": QueryType.AGGREGATION,
            "description": "Aggregation task"
        },
    ]
    
    passed = 0
    failed = 0
    
    for case in test_cases:
        query = case["query"]
        expected = case["expected"]
        description = case["description"]
        
        result = classifier.classify(query)
        actual = result.query_type
        
        if actual == expected:
            print(f"✅ PASSED: {description}")
            print(f"   Query: '{query[:60]}...'")
            print(f"   Type: {actual.value}")
            print(f"   Complexity: {result.complexity.value}")
            passed += 1
        else:
            print(f"❌ FAILED: {description}")
            print(f"   Query: '{query[:60]}...'")
            print(f"   Expected: {expected.value}, Got: {actual.value}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


async def test_skill_registry():
    """Test Skill Registry"""
    print_section("TEST 2: SKILL REGISTRY")
    
    registry = get_global_registry()
    
    # Get all skills
    all_skills = registry.list_skills()
    
    print(f"Total skills loaded: {len(all_skills)}")
    
    if len(all_skills) == 0:
        print("❌ FAILED: No skills loaded")
        return False
    
    # Just show total count, skip category grouping due to registry structure
    print(f"\nSkills loaded successfully")
    
    # Test key skills
    key_skills = [
        "search",
        "calculate",
        "analyze_data",
        "generate_code",
        "format_data"
    ]
    
    print(f"\nChecking key skills:")
    missing = []
    for skill_name in key_skills:
        if registry.has_skill(skill_name):
            print(f"  ✅ {skill_name}")
        else:
            print(f"  ❌ {skill_name} - MISSING")
            missing.append(skill_name)
    
    if missing:
        print(f"\n❌ FAILED: Missing skills: {missing}")
        return False
    else:
        print(f"\n✅ PASSED: All key skills available")
        return True


async def test_skill_execution():
    """Test Skill Execution"""
    print_section("TEST 3: SKILL EXECUTION")
    
    executor = SkillExecutor()
    
    test_cases = [
        {
            "skill": "calculate",
            "params": {"expression": "2 + 2"},
            "description": "Simple calculation"
        },
        {
            "skill": "format_data",
            "params": {
                "data": {"name": "John", "age": 30},
                "format": "json"
            },
            "description": "Data formatting"
        },
        {
            "skill": "analyze_data",
            "params": {
                "data": [1, 2, 3, 4, 5],
                "analysis_type": "statistics"
            },
            "description": "Data analysis"
        },
    ]
    
    passed = 0
    failed = 0
    
    for case in test_cases:
        skill_name = case["skill"]
        params = case["params"]
        description = case["description"]
        
        print(f"\nTesting: {description}")
        print(f"  Skill: {skill_name}")
        
        result = await executor.execute_skill(skill_name, params)
        
        if result.success:
            print(f"  ✅ PASSED")
            print(f"     Output: {str(result.output)[:100]}...")
            print(f"     Time: {result.execution_time:.3f}s")
            passed += 1
        else:
            print(f"  ❌ FAILED")
            print(f"     Error: {result.error}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


async def test_state_management():
    """Test State Management"""
    print_section("TEST 4: STATE MANAGEMENT")
    
    # Create state
    state = AgentState(query="Test query", session_id="test-001")
    
    checks = [
        (state.query == "Test query", "Query initialization"),
        (state.session_id == "test-001", "Session ID"),
        (state.iteration == 0, "Iteration counter"),
        (state.max_iterations == 3, "Max iterations"),
        (len(state.errors) == 0, "Error list"),
        (len(state.timings) == 0, "Timings dict"),
    ]
    
    passed = 0
    failed = 0
    
    for check, description in checks:
        if check:
            print(f"  ✅ {description}")
            passed += 1
        else:
            print(f"  ❌ {description}")
            failed += 1
    
    # Test state updates
    state.iteration += 1
    state.errors.append("Test error")
    state.timings["test_step"] = 1.5
    
    update_checks = [
        (state.iteration == 1, "Iteration increment"),
        (len(state.errors) == 1, "Error append"),
        (state.timings["test_step"] == 1.5, "Timing record"),
    ]
    
    for check, description in update_checks:
        if check:
            print(f"  ✅ {description}")
            passed += 1
        else:
            print(f"  ❌ {description}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


async def test_query_analysis_integration():
    """Test Query Analysis Integration"""
    print_section("TEST 5: QUERY ANALYSIS INTEGRATION")
    
    classifier = QueryClassifier()
    
    # Analyze a complex query
    query = "Compare the performance of supervised and unsupervised learning algorithms on image classification tasks"
    
    print(f"Query: {query}\n")
    
    result = classifier.classify(query)
    
    print(f"Analysis Results:")
    print(f"  Type: {result.query_type.value}")
    print(f"  Complexity: {result.complexity.value}")
    print(f"  Requires Reasoning: {result.requires_reasoning}")
    print(f"  Requires Refresh: {result.requires_refresh}")
    print(f"  Requires Graph: {result.requires_graph}")
    print(f"  Estimated Steps: {result.estimated_steps}")
    print(f"  Confidence: {result.confidence:.2f}")
    
    if result.keywords:
        print(f"  Keywords: {', '.join(result.keywords)}")
    
    # Validate results
    checks = [
        (result.query_type == QueryType.COMPARATIVE, "Query type detection"),
        (result.complexity.value in ["low", "medium", "high"], "Complexity assessment"),
        (result.requires_reasoning, "Reasoning requirement"),
        (result.estimated_steps > 1, "Step estimation"),
        (result.confidence > 0.0, "Confidence score"),
    ]
    
    passed = 0
    failed = 0
    
    print(f"\nValidation:")
    for check, description in checks:
        if check:
            print(f"  ✅ {description}")
            passed += 1
        else:
            print(f"  ❌ {description}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("SIMPLE SYSTEM INTEGRATION TEST SUITE")
    print("Testing Core Components Without Heavy Dependencies")
    print("="*80)
    
    tests = [
        ("Query Classifier", test_query_classifier),
        ("Skill Registry", test_skill_registry),
        ("Skill Execution", test_skill_execution),
        ("State Management", test_state_management),
        ("Query Analysis Integration", test_query_analysis_integration),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ TEST CRASHED: {name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Final summary
    print_section("FINAL SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nNext Steps:")
        print("  1. ✅ Core components are working correctly")
        print("  2. ✅ Skills system is functional")
        print("  3. ✅ Query classification is accurate")
        print("  4. 📋 Ready for integration with AdaptiveOrchestrator")
        print("  5. 📋 Ready for full system testing with real models")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
        print("\nPlease fix the failing tests before proceeding.")
    
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

