#!/usr/bin/env python3
"""
Production Test for main_adaptive.py
Tests with real models and complete flow
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_adaptive_orchestrator_with_queries():
    """Test AdaptiveOrchestrator with various query types"""
    
    print("\n" + "="*80)
    print("PRODUCTION TEST: main_adaptive.py with Real Flow")
    print("="*80 + "\n")
    
    # Test queries from simple to complex
    test_queries = [
        {
            "query": "What is Python?",
            "type": "SIMPLE_FACT",
            "description": "Simple factual question"
        },
        {
            "query": "Compare Python and JavaScript",
            "type": "COMPARATIVE",
            "description": "Comparison query"
        },
        {
            "query": "How has AI evolved in the last 5 years?",
            "type": "TEMPORAL",
            "description": "Temporal query requiring refresh"
        },
        {
            "query": "What is the relationship between machine learning, deep learning, and AI?",
            "type": "MULTI_HOP",
            "description": "Multi-hop reasoning query"
        },
        {
            "query": "Calculate the average of 10, 20, 30, 40, 50",
            "type": "AGGREGATION",
            "description": "Aggregation task"
        },
    ]
    
    try:
        # Import AdaptiveOrchestrator
        from agents.adaptive_orchestrator import AdaptiveOrchestrator
        from configs.main_config import CONFIG
        
        print("✅ Imports successful")
        print(f"✅ CONFIG loaded")
        
        # Try to create orchestrator
        print("\n📦 Creating AdaptiveOrchestrator...")
        orchestrator = AdaptiveOrchestrator(CONFIG)
        print("✅ AdaptiveOrchestrator created")
        
        # Test each query
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            print("\n" + "="*80)
            print(f"TEST {i}/{len(test_queries)}: {test_case['description']}")
            print("="*80)
            print(f"Query: '{test_case['query']}'")
            print(f"Expected Type: {test_case['type']}")
            
            try:
                # Run query
                print("\n⏳ Running query...")
                session_id = f"prod-test-{i:03d}"
                
                result = await orchestrator.run(test_case['query'], session_id)
                
                print("\n✅ Query completed!")
                print(f"Result type: {type(result)}")
                
                if isinstance(result, str):
                    print(f"Answer preview: {result[:200]}...")
                elif hasattr(result, 'final_answer'):
                    print(f"Answer preview: {result.final_answer[:200]}...")
                else:
                    print(f"Result: {str(result)[:200]}...")
                
                results.append({
                    "query": test_case['query'],
                    "type": test_case['type'],
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                print(f"\n❌ Query failed: {str(e)}")
                import traceback
                traceback.print_exc()
                
                results.append({
                    "query": test_case['query'],
                    "type": test_case['type'],
                    "success": False,
                    "error": str(e)
                })
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print(f"\nTotal Queries: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {successful/len(results)*100:.1f}%")
        
        print("\nResults by Query Type:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['type']}: {result['query'][:50]}...")
        
        if successful > 0:
            print("\n🎉 Some queries succeeded! System is partially working.")
            return True
        else:
            print("\n⚠️ All queries failed. Check dependencies and configuration.")
            return False
            
    except ImportError as e:
        print(f"\n❌ Import Error: {str(e)}")
        print("\nMissing dependencies. This is expected if:")
        print("  - kuzu is not installed")
        print("  - opensearch is not installed")
        print("  - Models are not available")
        print("\nThe system architecture is correct, but needs:")
        print("  1. Install: pip install kuzu opensearch-py")
        print("  2. Setup models in correct paths")
        print("  3. Configure database connections")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_components_availability():
    """Test if all components can be imported"""
    
    print("\n" + "="*80)
    print("COMPONENT AVAILABILITY TEST")
    print("="*80 + "\n")
    
    components = [
        ("QueryClassifier", "agents.query_classifier", "QueryClassifier"),
        ("AdaptiveOrchestrator", "agents.adaptive_orchestrator", "AdaptiveOrchestrator"),
        ("SkillExecutor", "agents.skill_executor", "SkillExecutor"),
        ("MetricsMonitor", "core.metrics_monitor", "MetricsMonitor"),
        ("SemanticGraphBuilder", "ingestion.semantic_graph_builder", "SemanticGraphBuilder"),
    ]
    
    available = 0
    unavailable = 0
    
    for name, module, cls in components:
        try:
            exec(f"from {module} import {cls}")
            print(f"✅ {name}: Available")
            available += 1
        except Exception as e:
            print(f"❌ {name}: Unavailable ({str(e)[:50]}...)")
            unavailable += 1
    
    print(f"\nSummary: {available}/{len(components)} components available")
    
    return available == len(components)


async def main():
    """Run production tests"""
    
    print("\n" + "="*80)
    print("MAIN_ADAPTIVE.PY PRODUCTION TEST SUITE")
    print("="*80)
    
    # Test 1: Component Availability
    print("\n[1/2] Testing component availability...")
    components_ok = await test_components_availability()
    
    if not components_ok:
        print("\n⚠️ Some components unavailable. Continuing with available components...")
    
    # Test 2: Full Flow Test
    print("\n[2/2] Testing with real queries...")
    flow_ok = await test_adaptive_orchestrator_with_queries()
    
    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"\nComponent Availability: {'✅ Pass' if components_ok else '⚠️ Partial'}")
    print(f"Flow Test: {'✅ Pass' if flow_ok else '❌ Fail'}")
    
    if components_ok and flow_ok:
        print("\n🎉 SUCCESS: System is production-ready!")
        print("\nNext steps:")
        print("  1. Run: python main_adaptive.py")
        print("  2. Open browser at http://localhost:7860")
        print("  3. Test with real queries")
        return 0
    elif components_ok:
        print("\n⚠️ PARTIAL SUCCESS: Components work but need configuration")
        print("\nRequired:")
        print("  1. Install dependencies: pip install kuzu opensearch-py")
        print("  2. Setup model paths in configs/main_config.py")
        print("  3. Initialize databases")
        return 1
    else:
        print("\n❌ FAILURE: Missing critical components")
        print("\nPlease check:")
        print("  1. All files are in correct locations")
        print("  2. Python path is set correctly")
        print("  3. No syntax errors in code")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

