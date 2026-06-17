"""
Test All Skills Loading and Basic Functionality
"""

import asyncio
from agents.skill_executor import SkillExecutor, get_global_registry


async def test_skills_loading():
    """تست بارگذاری تمام skills"""
    print("\n" + "="*80)
    print("TESTING ALL SKILLS LOADING")
    print("="*80)
    
    # بارگذاری skills
    import agents  # این باعث بارگذاری خودکار می‌شود
    
    registry = get_global_registry()
    skills = registry.list_skills()
    
    print(f"\n✅ Total skills loaded: {len(skills)}")
    print("\nSkills by category:")
    
    # دسته‌بندی skills
    categories = {}
    for skill in skills:
        skill_type = skill.get('type', 'unknown')
        if skill_type not in categories:
            categories[skill_type] = []
        categories[skill_type].append(skill['name'])
    
    for category, skill_names in sorted(categories.items()):
        print(f"\n{category.upper()}:")
        for name in sorted(skill_names):
            print(f"  - {name}")
    
    return len(skills)


async def test_basic_skills():
    """تست عملکرد پایه چند skill"""
    print("\n" + "="*80)
    print("TESTING BASIC SKILL FUNCTIONALITY")
    print("="*80)
    
    executor = SkillExecutor(get_global_registry())
    
    tests = [
        {
            "name": "calculate",
            "args": {"expression": "2 + 2"},
            "expected_key": "result"
        },
        {
            "name": "analyze_data",
            "args": {"data": {"test": [1, 2, 3]}, "analysis_type": "summary"},
            "expected_key": "type"
        },
        {
            "name": "format_data",
            "args": {"data": {"key": "value"}, "from_format": "dict", "to_format": "json"},
            "expected_key": "success"
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nTesting: {test['name']}")
            result = await executor.execute_skill(test['name'], test['args'])
            
            if result.success and test['expected_key'] in result.output:
                print(f"  ✅ PASSED")
                passed += 1
            else:
                print(f"  ❌ FAILED: {result.error if not result.success else 'Missing expected key'}")
                failed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*80}")
    
    return passed, failed


async def test_specialized_skills():
    """تست skills تخصصی"""
    print("\n" + "="*80)
    print("TESTING SPECIALIZED SKILLS")
    print("="*80)
    
    executor = SkillExecutor(get_global_registry())
    
    tests = [
        {
            "name": "analyze_code_quality",
            "args": {"code": "def test():\n    pass", "language": "python"},
            "category": "Code"
        },
        {
            "name": "calculate_financial_metrics",
            "args": {"data": {"profit": 100, "investment": 500}, "metrics": ["roi"]},
            "category": "Finance"
        },
        {
            "name": "process_sensor_data",
            "args": {"data": [{"value": 25}, {"value": 30}], "sensor_type": "temperature"},
            "category": "Sensor"
        },
        {
            "name": "compute_vector_similarity",
            "args": {"vector1": [1, 2, 3], "vector2": [1, 2, 3], "method": "cosine"},
            "category": "Vector"
        },
        {
            "name": "generate_color_palette",
            "args": {"base_color": "#3498db", "scheme": "complementary", "count": 5},
            "category": "Design"
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nTesting [{test['category']}]: {test['name']}")
            result = await executor.execute_skill(test['name'], test['args'])
            
            if result.success:
                print(f"  ✅ PASSED")
                print(f"     Output keys: {list(result.output.keys())[:5]}")
                passed += 1
            else:
                print(f"  ❌ FAILED: {result.error}")
                failed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*80}")
    
    return passed, failed


async def main():
    """اجرای تمام تست‌ها"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SKILLS TEST SUITE")
    print("="*80)
    
    # تست 1: بارگذاری
    total_skills = await test_skills_loading()
    
    # تست 2: عملکرد پایه
    basic_passed, basic_failed = await test_basic_skills()
    
    # تست 3: skills تخصصی
    spec_passed, spec_failed = await test_specialized_skills()
    
    # خلاصه نهایی
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total Skills Loaded: {total_skills}")
    print(f"Basic Skills Tests: {basic_passed} passed, {basic_failed} failed")
    print(f"Specialized Skills Tests: {spec_passed} passed, {spec_failed} failed")
    print(f"\nOverall: {basic_passed + spec_passed} passed, {basic_failed + spec_failed} failed")
    
    if basic_failed + spec_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {basic_failed + spec_failed} test(s) failed")
    
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

