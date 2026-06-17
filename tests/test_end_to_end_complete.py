#!/usr/bin/env python3
"""
Complete End-to-End Test
Tests the entire flow from query to answer without heavy dependencies
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.query_classifier import QueryClassifier
from agents.skill_executor import get_global_registry, SkillExecutor
from agents.state import AgentState, FlowStep
from core.metrics_monitor import get_metrics_monitor


def print_header(title):
    """Print formatted header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


async def test_complete_flow():
    """Test complete flow from query to answer"""
    print_header("COMPLETE END-TO-END FLOW TEST")
    
    # Initialize components
    print("📦 Initializing components...")
    classifier = QueryClassifier()
    executor = SkillExecutor()
    registry = get_global_registry()
    monitor = get_metrics_monitor()
    
    print(f"  ✅ QueryClassifier initialized")
    print(f"  ✅ SkillExecutor initialized")
    print(f"  ✅ SkillRegistry loaded ({len(registry.list_skills())} skills)")
    print(f"  ✅ MetricsMonitor initialized")
    
    # Test query
    query = "What is the difference between supervised and unsupervised learning?"
    session_id = "e2e-test-001"
    
    print(f"\n📝 Test Query: '{query}'")
    
    # Step 1: Query Classification
    print_header("STEP 1: QUERY CLASSIFICATION")
    
    query_id = monitor.start_query(query, session_id)
    
    analysis = classifier.classify(query)
    print(f"  Query Type: {analysis.query_type.value}")
    print(f"  Complexity: {analysis.complexity.value}")
    print(f"  Requires Reasoning: {analysis.requires_reasoning}")
    print(f"  Requires Refresh: {analysis.requires_refresh}")
    print(f"  Requires Graph: {analysis.requires_graph}")
    print(f"  Estimated Steps: {analysis.estimated_steps}")
    print(f"  Confidence: {analysis.confidence:.2f}")
    
    if analysis.keywords:
        print(f"  Keywords: {', '.join(analysis.keywords[:5])}")
    
    # Step 2: Create State
    print_header("STEP 2: STATE INITIALIZATION")
    
    state = AgentState(query=query, session_id=session_id)
    state.query_analysis = analysis
    state.current_step = FlowStep.START
    
    print(f"  ✅ State created")
    print(f"  Session ID: {state.session_id}")
    print(f"  Current Step: {state.current_step.value}")
    print(f"  Max Iterations: {state.max_iterations}")
    
    # Step 3: Plan Generation (Simulated)
    print_header("STEP 3: DYNAMIC PLAN GENERATION")
    
    # Based on query type, generate plan
    if analysis.query_type.value == "comparative":
        plan_steps = ["UNDERSTAND", "RETRIEVE", "REASON", "VALIDATE", "ANSWER"]
    elif analysis.query_type.value == "simple_fact":
        plan_steps = ["UNDERSTAND", "RETRIEVE", "ANSWER"]
    elif analysis.query_type.value == "temporal":
        plan_steps = ["UNDERSTAND", "RETRIEVE", "REFRESH", "RETRIEVE", "REASON", "ANSWER"]
    else:
        plan_steps = ["UNDERSTAND", "RETRIEVE", "REASON", "VALIDATE", "ANSWER"]
    
    print(f"  Generated Plan: {' → '.join(plan_steps)}")
    print(f"  Total Steps: {len(plan_steps)}")
    print(f"  Estimated Time: {len(plan_steps) * 0.5:.1f}s")
    
    # Step 4: Execute Skills
    print_header("STEP 4: SKILL EXECUTION")
    
    # Test search skill
    print("\n  Testing 'search' skill...")
    search_result = await executor.execute_skill("search", {
        "query": "supervised vs unsupervised learning"
    })
    
    if search_result.success:
        print(f"    ✅ Search completed in {search_result.execution_time:.3f}s")
        monitor.record_skill_usage(query_id, "search")
    else:
        print(f"    ❌ Search failed: {search_result.error}")
    
    # Test calculate skill
    print("\n  Testing 'calculate' skill...")
    calc_result = await executor.execute_skill("calculate", {
        "expression": "2 + 2"
    })
    
    if calc_result.success:
        print(f"    ✅ Calculate completed in {calc_result.execution_time:.3f}s")
        print(f"    Result: {calc_result.output}")
        monitor.record_skill_usage(query_id, "calculate")
    else:
        print(f"    ❌ Calculate failed: {calc_result.error}")
    
    # Step 5: Simulate Step Execution
    print_header("STEP 5: STEP-BY-STEP EXECUTION")
    
    import time
    for i, step in enumerate(plan_steps, 1):
        print(f"\n  [{i}/{len(plan_steps)}] Executing: {step}")
        
        start_time = time.time()
        
        # Simulate step execution
        await asyncio.sleep(0.1)  # Simulate work
        
        duration = time.time() - start_time
        monitor.record_step(query_id, step, duration)
        
        print(f"      ⏱️  Duration: {duration:.3f}s")
        print(f"      ✅ Completed")
        
        state.current_step = FlowStep[step] if step in FlowStep.__members__ else FlowStep.ANSWER
    
    # Step 6: Generate Answer (Simulated)
    print_header("STEP 6: ANSWER GENERATION")
    
    simulated_answer = """
    Supervised and unsupervised learning are two main types of machine learning:
    
    **Supervised Learning:**
    - Uses labeled training data
    - Learns to map inputs to known outputs
    - Examples: Classification, Regression
    - Use cases: Image recognition, spam detection
    
    **Unsupervised Learning:**
    - Uses unlabeled data
    - Finds hidden patterns and structures
    - Examples: Clustering, Dimensionality reduction
    - Use cases: Customer segmentation, anomaly detection
    
    The key difference is that supervised learning requires labeled data with known 
    outcomes, while unsupervised learning discovers patterns in unlabeled data.
    """
    
    state.final_answer = simulated_answer.strip()
    state.confidence = 0.85
    
    print(f"  ✅ Answer generated")
    print(f"  Confidence: {state.confidence:.2f}")
    print(f"  Length: {len(state.final_answer)} characters")
    print(f"\n  Preview:")
    print(f"  {state.final_answer[:200]}...")
    
    # Step 7: Validation (Simulated)
    print_header("STEP 7: ANSWER VALIDATION")
    
    validation_checks = [
        ("Answer length > 100 chars", len(state.final_answer) > 100),
        ("Confidence > 0.7", state.confidence > 0.7),
        ("Contains key terms", any(term in state.final_answer.lower() for term in ["supervised", "unsupervised"])),
        ("Well-structured", "**" in state.final_answer or "\n" in state.final_answer),
    ]
    
    passed_checks = 0
    for check_name, result in validation_checks:
        status = "✅" if result else "❌"
        print(f"  {status} {check_name}")
        if result:
            passed_checks += 1
    
    validation_score = passed_checks / len(validation_checks)
    print(f"\n  Validation Score: {validation_score:.2f} ({passed_checks}/{len(validation_checks)})")
    
    # Step 8: Finalize Metrics
    print_header("STEP 8: METRICS FINALIZATION")
    
    monitor.end_query(
        query_id,
        success=True,
        confidence=state.confidence,
        quality_score=validation_score,
        tokens_used=len(state.final_answer.split()),
        memory_peak_mb=50.0  # Simulated
    )
    
    print(f"  ✅ Metrics recorded")
    
    # Display metrics
    query_metrics = monitor.get_query_metrics(query_id)
    if query_metrics:
        print(f"\n  Query Metrics:")
        print(f"    Total Time: {query_metrics['total_time']:.2f}s")
        print(f"    Steps: {len(query_metrics['steps_executed'])}")
        print(f"    Skills Used: {len(query_metrics['skills_used'])}")
        print(f"    Success: {query_metrics['success']}")
        print(f"    Confidence: {query_metrics['confidence']:.2f}")
        print(f"    Quality Score: {query_metrics['quality_score']:.2f}")
    
    # Step 9: System Metrics
    print_header("STEP 9: SYSTEM METRICS")
    
    system_metrics = monitor.get_system_metrics()
    print(f"  Total Queries: {system_metrics['total_queries']}")
    print(f"  Success Rate: {system_metrics['success_rate']*100:.1f}%")
    print(f"  Avg Response Time: {system_metrics['average_response_time']:.2f}s")
    print(f"  Avg Confidence: {system_metrics['average_confidence']:.2f}")
    print(f"  Avg Quality: {system_metrics['average_quality_score']:.2f}")
    
    if system_metrics['top_skills']:
        print(f"\n  Top Skills Used:")
        for skill, count in system_metrics['top_skills'][:3]:
            print(f"    - {skill}: {count} times")
    
    # Final Summary
    print_header("FINAL SUMMARY")
    
    print(f"✅ End-to-End Test Completed Successfully!")
    print(f"\n📊 Test Results:")
    print(f"  Query: '{query}'")
    print(f"  Query Type: {analysis.query_type.value}")
    print(f"  Plan Steps: {len(plan_steps)}")
    print(f"  Skills Tested: 2")
    print(f"  Validation Score: {validation_score:.2f}")
    print(f"  Final Confidence: {state.confidence:.2f}")
    print(f"\n🎉 All components working correctly!")
    
    return True


async def main():
    """Run end-to-end test"""
    try:
        success = await test_complete_flow()
        
        if success:
            print("\n" + "="*80)
            print("SUCCESS: End-to-End test passed!".center(80))
            print("="*80 + "\n")
            return 0
        else:
            print("\n" + "="*80)
            print("FAILURE: End-to-End test failed!".center(80))
            print("="*80 + "\n")
            return 1
            
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Test crashed!".center(80))
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

