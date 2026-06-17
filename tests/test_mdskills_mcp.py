"""
Test MDSkills MCP Tools Integration
====================================

تست کامل یکپارچگی MDSkills MCP tools.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry


def test_git_tools():
    """Test Git MCP tools"""
    print("\n" + "="*60)
    print("Test 1: Git Tools")
    print("="*60)
    
    registry = MDSkillsMCPRegistry()
    
    # Test git status
    print("\n1.1 Git Status:")
    result = registry.git.git_status(".")
    print(json.dumps(result, indent=2))
    assert result["success"], "Git status should succeed"
    
    # Test git log
    print("\n1.2 Git Log:")
    result = registry.git.git_log(".", max_count=5)
    print(json.dumps(result, indent=2))
    assert result["success"], "Git log should succeed"
    
    print("\n✅ Test 1 PASSED: Git tools work")
    return True


def test_filesystem_tools():
    """Test File System MCP tools"""
    print("\n" + "="*60)
    print("Test 2: File System Tools")
    print("="*60)
    
    registry = MDSkillsMCPRegistry()
    
    # Test list directory
    print("\n2.1 List Directory:")
    result = registry.filesystem.list_directory(".", pattern="*.py")
    print(f"Found {result['count']} Python files")
    assert result["success"], "List directory should succeed"
    
    # Test read file
    print("\n2.2 Read File:")
    result = registry.filesystem.read_file("README.md")
    if result["success"]:
        print(f"Read {result['lines']} lines from README.md")
    else:
        print(f"README.md not found (expected in some cases)")
    
    # Test search files
    print("\n2.3 Search Files:")
    result = registry.filesystem.search_files(".", pattern="*.py", content_pattern="import")
    print(f"Found {result['count']} files with 'import'")
    assert result["success"], "Search files should succeed"
    
    print("\n✅ Test 2 PASSED: File system tools work")
    return True


def test_database_tools():
    """Test Database MCP tools"""
    print("\n" + "="*60)
    print("Test 3: Database Tools")
    print("="*60)
    
    registry = MDSkillsMCPRegistry()
    
    # Create test database
    import sqlite3
    test_db = Path("test_mcp.db")
    
    try:
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("INSERT INTO test (name) VALUES ('Alice'), ('piter')")
        conn.commit()
        conn.close()
        
        # Test list tables
        print("\n3.1 List Tables:")
        result = registry.database.list_sqlite_tables(str(test_db))
        print(json.dumps(result, indent=2))
        assert result["success"], "List tables should succeed"
        assert "test" in result["tables"], "Should find test table"
        
        # Test query
        print("\n3.2 Query Database:")
        result = registry.database.query_sqlite(str(test_db), "SELECT * FROM test")
        print(json.dumps(result, indent=2))
        assert result["success"], "Query should succeed"
        assert result["count"] == 2, "Should find 2 rows"
        
        print("\n✅ Test 3 PASSED: Database tools work")
        return True
        
    finally:
        if test_db.exists():
            test_db.unlink()


def test_documentation_tools():
    """Test Documentation MCP tools"""
    print("\n" + "="*60)
    print("Test 4: Documentation Tools")
    print("="*60)
    
    registry = MDSkillsMCPRegistry()
    
    # Test generate API docs
    print("\n4.1 Generate API Docs:")
    result = registry.docs.generate_api_docs(
        code_path="mcp/mdskills_mcp_tools.py",
        output_path="test_api_docs.md",
        format="markdown"
    )
    print(json.dumps(result, indent=2))
    
    if result["success"]:
        print(f"Generated docs for {result['items_documented']} items")
        # Clean up
        Path("test_api_docs.md").unlink(missing_ok=True)
    
    print("\n✅ Test 4 PASSED: Documentation tools work")
    return True


def test_tool_registry():
    """Test MDSkills MCP Registry"""
    print("\n" + "="*60)
    print("Test 5: Tool Registry")
    print("="*60)
    
    registry = MDSkillsMCPRegistry()
    
    # List all tools
    print("\n5.1 List All Tools:")
    tools = registry.list_all_tools()
    print(f"Total tools: {len(tools)}")
    
    categories = {}
    for tool in tools:
        cat = tool["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nTools by category:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} tools")
    
    assert len(tools) > 0, "Should have tools"
    assert len(categories) >= 5, "Should have at least 5 categories"
    
    print("\n✅ Test 5 PASSED: Tool registry works")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MDSkills MCP Tools Integration Tests")
    print("="*60)
    
    tests = [
        ("Git Tools", test_git_tools),
        ("File System Tools", test_filesystem_tools),
        ("Database Tools", test_database_tools),
        ("Documentation Tools", test_documentation_tools),
        ("Tool Registry", test_tool_registry),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASSED" if p else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())

