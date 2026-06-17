#!/usr/bin/env python3
"""
End-to-End Integration Test for Complete System
Tests the full flow from query to answer using AdaptiveOrchestrator
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.adaptive_orchestrator import AdaptiveOrchestrator
from agents.query_classifier import QueryClassifier
from agents.state import AgentState
from agents.skill_executor import get_global_registry
from core.logger import setup_logger

logger = setup_logger(__name__)


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.confidence_threshold = 0.65
        self.max_iterations = 3
        
    def get(self, key, default=None):
        return getattr(self, key, default)


class MockPhi4:
    """Mock Phi4 client for testing"""
    
    async def understand_and_plan(self, query: str, history: list = None):
        """Mock understand and plan"""
        return {
            "intent": "information_retrieval",
            "entities": [
                {"type": "TOPIC", "value": "machine learning", "confidence": 0.9}
            ],
            "sub_questions": [
                "What is machine learning?",
                "What are the types of machine learning?"
            ],
            "steps": [
                {
                    "id": 1,
                    "action": "search",
                    "description": "Search for machine learning information",
                    "depends_on": []
                },
                {
                    "id": 2,
                    "action": "analyze_data",
                    "description": "Analyze search results",
                    "depends_on": [1]
                }
            ],
            "strategy": "multi_step_retrieval",
            "tool_calls": [
                {
                    "name": "search",
                    "arguments": {"query": "machine learning basics"}
                }
            ],
            "quality_score": 0.85,
            "processing_time": 0.5,
            "metadata": {}
        }
    
    async def detect_gaps(self, query: str, context: str):
        """Mock gap detection"""
        return {
            "sufficient": True,
            "confidence": 0.8,
            "missing_info": [],
            "search_queries": []
        }


class MockRetriever:
    """Mock retriever for testing"""
    
    async def startup(self):
        pass
    
    async def retrieve(self, state: AgentState, query: str = None):
        """Mock retrieval"""
        state.retrieval = type('obj', (object,), {
            'confidence': 0.75,
            'vector_chunks': [
                {"content": "Machine learning is a subset of AI", "score": 0.9},
                {"content": "Types include supervised and unsupervised", "score": 0.85}
            ],
            'context': "Machine learning is a subset of AI. Types include supervised and unsupervised learning."
        })()
        return state


class MockReasoner:
    """Mock reasoner for testing"""
    
    async def reason(self, state: AgentState):
        """Mock reasoning"""
        state.reasoning = {
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "confidence": 0.85,
            "reasoning_steps": [
                "Analyzed query intent",
                "Retrieved relevant information",
                "Synthesized answer from context"
            ]
        }
        return state


class MockValidator:
    """Mock validator for testing"""
    
    async def validate(self, state: AgentState):
        """Mock validation"""
        return {
            "score": 0.85,
            "answer": state.reasoning.get("answer", "No answer generated"),
            "issues": [],
            "suggestions": []
        }


async def test_query_classification():
    """Test 1: Query Classification"""
    print("\n" + "="*80)
    print("TEST 1: QUERY CLASSIFICATION")
    print("="*80)
    
    classifier = QueryClassifier()
    
    test_queries = [
        ("What is machine learning?", "SIMPLE_FACT"),
        ("Compare supervised and unsupervised learning", "COMPARATIVE"),
        ("How has AI evolved over the past decade?", "TEMPORAL"),
        ("Write a creative story about robots", "CREATIVE"),
        ("What are the connections between AI, ML, and DL?", "MULTI_HOP"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_type in test_queries:
        result = await classifier.classify(query)
        actual_type = result.query_type.value
        
        if actual_type == expected_type:
            print(f"✅ PASSED: '{query[:50]}...' -> {actual_type}")
            passed += 1
        else:
            print(f"❌ FAILED: '{query[:50]}...' -> Expected {expected_type}, got {actual_type}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


async def test_skill_loading():
    """Test 2: Skill Loading and Registry"""
    print("\n" + "="*80)
    print("TEST 2: SKILL LOADING AND REGISTRY")
    print("="*80)
    
    registry = get_global_registry()
    
    # Check if skills are loaded
    all_skills = registry.list_skills()
    
    if len(all_skills) > 0:
        print(f"✅ PASSED: {len(all_skills)} skills loaded")
        
        # Check for key skills
        key_skills = ["search", "analyze_data", "calculate", "generate_code"]
        missing = []
        
        for skill in key_skills:
            if registry.has_skill(skill):
                print(f"  ✅ {skill} available")
            else:
                print(f"  ❌ {skill} missing")
                missing.append(skill)
        
        if missing:
            print(f"\n❌ FAILED: Missing skills: {missing}")
            return False
        else:
            print(f"\n✅ All key skills available")
            return True
    else:
        print("❌ FAILED: No skills loaded")
        return False


async def test_adaptive_orchestrator_initialization():
    """Test 3: AdaptiveOrchestrator Initialization"""
    print("\n" + "="*80)
    print("TEST 3: ADAPTIVE ORCHESTRATOR INITIALIZATION")
    print("="*80)
    
    try:
        config = MockConfig()
        orchestrator = AdaptiveOrchestrator(config)
        
        # Check components
        if orchestrator.classifier is None:
            print("❌ FAILED: QueryClassifier not initialized")
            return False
        
        if orchestrator.skill_executor is None:
            print("❌ FAILED: SkillExecutor not initialized")
            return False
        
        print("✅ PASSED: AdaptiveOrchestrator initialized successfully")
        print(f"  - QueryClassifier: ✅")
        print(f"  - SkillExecutor: ✅")
        print(f"  - Config: ✅")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False


async def test_understand_and_analyze_step():
    """Test 4: Understand and Analyze Step"""
    print("\n" + "="*80)
    print("TEST 4: UNDERSTAND AND ANALYZE STEP")
    print("="*80)
    
    try:
        config = MockConfig()
        orchestrator = AdaptiveOrchestrator(config)
        
        # Mock phi4
        orchestrator.config = {"phi4_mini_factory": lambda: MockPhi4()}
        
        state = AgentState(query="What is machine learning?", session_id="test-001")
        
        # Execute understand step
        next_step = await orchestrator._step_understand_and_analyze(state)
        
        # Check results
        checks = [
            (state.query_analysis is not None, "Query analysis"),
            (state.intent is not None, "Intent detection"),
            (len(state.extracted_entities) > 0, "Entity extraction"),
            (len(state.sub_questions) > 0, "Sub-question generation"),
            (len(state.plan_steps) > 0, "Plan generation"),
        ]
        
        passed = 0
        failed = 0
        
        for check, name in checks:
            if check:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name}")
                failed += 1
        
        if failed == 0:
            print(f"\n✅ PASSED: All checks passed ({passed}/{len(checks)})")
            return True
        else:
            print(f"\n❌ FAILED: {failed} checks failed")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_dynamic_plan_generation():
    """Test 5: Dynamic Plan Generation"""
    print("\n" + "="*80)
    print("TEST 5: DYNAMIC PLAN GENERATION")
    print("="*80)
    
    try:
        config = MockConfig()
        orchestrator = AdaptiveOrchestrator(config)
        
        # Create state with query analysis
        state = AgentState(query="Compare supervised and unsupervised learning", session_id="test-002")
        state.query_analysis = type('obj', (object,), {
            'query_type': type('obj', (object,), {'value': 'COMPARATIVE'})(),
            'complexity': 0.7,
            'requires_reasoning': True,
            'requires_external_data': False
        })()
        
        # Generate plan
        plan = await orchestrator._generate_dynamic_plan(state)
        
        # Check plan
        checks = [
            (plan is not None, "Plan generated"),
            (len(plan.steps) > 0, "Plan has steps"),
            (plan.estimated_time > 0, "Estimated time calculated"),
        ]
        
        passed = 0
        failed = 0
        
        for check, name in checks:
            if check:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name}")
                failed += 1
        
        if plan:
            print(f"\n  Plan details:")
            print(f"    - Steps: {len(plan.steps)}")
            print(f"    - Estimated time: {plan.estimated_time:.2f}s")
            print(f"    - Can skip: {', '.join(plan.can_skip) if plan.can_skip else 'None'}")
        
        if failed == 0:
            print(f"\n✅ PASSED: All checks passed ({passed}/{len(checks)})")
            return True
        else:
            print(f"\n❌ FAILED: {failed} checks failed")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_flow_simple_query():
    """Test 6: Full Flow with Simple Query"""
    print("\n" + "="*80)
    print("TEST 6: FULL FLOW WITH SIMPLE QUERY")
    print("="*80)
    
    try:
        config = MockConfig()
        orchestrator = AdaptiveOrchestrator(config)
        
        # Mock dependencies
        orchestrator.retriever = MockRetriever()
        orchestrator.reasoner = MockReasoner()
        orchestrator.validator = MockValidator()
        
        # Mock phi4 factory
        class MockPhi4Factory:
            def __call__(self):
                return self
            async def __aenter__(self):
                return MockPhi4()
            async def __aexit__(self, *args):
                pass
        
        orchestrator.config = {
            "phi4_mini_factory": MockPhi4Factory(),
            "confidence_threshold": 0.65
        }
        
        # Run full flow
        state = AgentState(query="What is machine learning?", session_id="test-003")
        
        print("\n  Executing full flow...")
        result = await orchestrator.run(state.query, state.session_id)
        
        # Check results
        checks = [
            (result is not None, "Result generated"),
            (isinstance(result, str) or hasattr(result, 'final_answer'), "Valid result format"),
        ]
        
        passed = 0
        failed = 0
        
        for check, name in checks:
            if check:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name}")
                failed += 1
        
        if failed == 0:
            print(f"\n✅ PASSED: Full flow completed successfully")
            if isinstance(result, str):
                print(f"\n  Answer: {result[:200]}...")
            return True
        else:
            print(f"\n❌ FAILED: {failed} checks failed")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("END-TO-END INTEGRATION TEST SUITE")
    print("Testing Complete System with AdaptiveOrchestrator")
    print("="*80)
    
    tests = [
        ("Query Classification", test_query_classification),
        ("Skill Loading", test_skill_loading),
        ("Orchestrator Initialization", test_adaptive_orchestrator_initialization),
        ("Understand and Analyze", test_understand_and_analyze_step),
        ("Dynamic Plan Generation", test_dynamic_plan_generation),
        ("Full Flow", test_full_flow_simple_query),
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
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")
    
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

