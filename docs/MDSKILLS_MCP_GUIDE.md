# راهنمای MDSkills MCP Tools

## 📋 فهرست مطالب
1. [معرفی](#معرفی)
2. [نصب و راه‌اندازی](#نصب-و-راه‌اندازی)
3. [دسته‌بندی Tools](#دسته‌بندی-tools)
4. [استفاده](#استفاده)
5. [مثال‌های کاربردی](#مثال‌های-کاربردی)
6. [یکپارچگی با MCP Server](#یکپارچگی-با-mcp-server)

---

## معرفی

این سیستم بر اساس **MDSkills.ai** پیاده‌سازی شده و شامل 15+ MCP tool در 6 دسته است:

### منابع:
- 🔗 [MDSkills.ai MCP Servers](https://www.mdskills.ai/mcp-servers)
- 🔗 [MDSkills.ai Developer Tools](https://www.mdskills.ai/use-cases/developer-tools)

### ویژگی‌های کلیدی:
- ✅ **15+ Tools**: Git, GitHub, File System, Database, Web, Documentation
- ✅ **Production Ready**: تست شده و آماده استفاده
- ✅ **MCP Compatible**: سازگار با Claude Desktop, Cursor, و سایر کلاینت‌ها
- ✅ **Async Support**: پشتیبانی کامل از async/await
- ✅ **Error Handling**: مدیریت خطای جامع

---

## نصب و راه‌اندازی

### پیش‌نیازها:
```bash
pip install aiohttp beautifulsoup4
```

### استفاده مستقیم:
```python
from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry

# Initialize registry
registry = MDSkillsMCPRegistry()

# List all available tools
tools = registry.list_all_tools()
print(f"Available tools: {len(tools)}")
```

---

## دسته‌بندی Tools

### 1. Git Tools (4 tools)

#### `git_status`
وضعیت repository را نمایش می‌دهد.

```python
result = registry.git.git_status(".")
# Output: {"success": True, "files": [...], "clean": False}
```

#### `git_log`
تاریخچه commit‌ها را نمایش می‌دهد.

```python
result = registry.git.git_log(".", max_count=10)
# Output: {"success": True, "commits": [...]}
```

#### `git_diff`
تغییرات را نمایش می‌دهد.

```python
result = registry.git.git_diff(".", file_path="main.py")
# Output: {"success": True, "diff": "..."}
```

#### `git_commit`
یک commit جدید ایجاد می‌کند.

```python
result = registry.git.git_commit(".", message="Update feature", files=["main.py"])
# Output: {"success": True, "message": "Update feature"}
```

---

### 2. GitHub Tools (2 tools)

#### `search_repositories`
جستجو در GitHub repositories.

```python
result = await registry.github.search_repositories("machine learning", max_results=10)
# Output: {"success": True, "repositories": [...]}
```

#### `get_repository_info`
اطلاعات یک repository را دریافت می‌کند.

```python
result = await registry.github.get_repository_info("openai", "gpt-3")
# Output: {"success": True, "name": "gpt-3", "stars": 1000, ...}
```

---

### 3. File System Tools (4 tools)

#### `read_file`
محتوای فایل را می‌خواند.

```python
result = registry.filesystem.read_file("README.md")
# Output: {"success": True, "content": "...", "lines": 100}
```

#### `write_file`
محتوا را در فایل می‌نویسد.

```python
result = registry.filesystem.write_file("output.txt", "Hello World")
# Output: {"success": True, "file_path": "output.txt"}
```

#### `list_directory`
محتویات دایرکتوری را لیست می‌کند.

```python
result = registry.filesystem.list_directory(".", recursive=True, pattern="*.py")
# Output: {"success": True, "items": [...], "count": 50}
```

#### `search_files`
فایل‌ها را بر اساس نام و محتوا جستجو می‌کند.

```python
result = registry.filesystem.search_files(".", pattern="*.py", content_pattern="import")
# Output: {"success": True, "results": [...], "count": 30}
```

---

### 4. Database Tools (2 tools)

#### `query_sqlite`
کوئری SQLite را اجرا می‌کند.

```python
result = registry.database.query_sqlite("data.db", "SELECT * FROM users")
# Output: {"success": True, "rows": [...], "count": 10}
```

#### `list_sqlite_tables`
جداول SQLite را لیست می‌کند.

```python
result = registry.database.list_sqlite_tables("data.db")
# Output: {"success": True, "tables": ["users", "posts"], "count": 2}
```

---

### 5. Web Tools (2 tools)

#### `fetch_url`
محتوای URL را دریافت می‌کند.

```python
result = await registry.web.fetch_url("https://example.com")
# Output: {"success": True, "content": "...", "status": 200}
```

#### `scrape_webpage`
صفحه وب را scrape می‌کند.

```python
result = await registry.web.scrape_webpage("https://example.com", selector=".title")
# Output: {"success": True, "content": [...], "title": "Example"}
```

---

### 6. Documentation Tools (1 tool)

#### `generate_api_docs`
مستندات API را از کد تولید می‌کند.

```python
result = registry.docs.generate_api_docs(
    code_path="src/",
    output_path="docs/api.md",
    format="markdown"
)
# Output: {"success": True, "items_documented": 50}
```

---

## استفاده

### روش 1: استفاده مستقیم

```python
from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry

# Initialize
registry = MDSkillsMCPRegistry()

# Git operations
status = registry.git.git_status(".")
print(f"Modified files: {len(status['files'])}")

# File operations
content = registry.filesystem.read_file("README.md")
print(f"README has {content['lines']} lines")

# Database operations
tables = registry.database.list_sqlite_tables("data.db")
print(f"Database has {tables['count']} tables")
```

### روش 2: استفاده از Registry

```python
import asyncio
from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry

async def main():
    registry = MDSkillsMCPRegistry()
    
    # Execute tool by category and name
    result = await registry.execute_tool(
        category="GitHub",
        tool_name="search_repositories",
        query="python machine learning",
        max_results=5
    )
    
    print(result)

asyncio.run(main())
```

---

## مثال‌های کاربردی

### مثال 1: تحلیل Repository

```python
from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry
import json

registry = MDSkillsMCPRegistry()

# 1. بررسی وضعیت Git
status = registry.git.git_status(".")
print(f"📊 Repository Status:")
print(f"  - Modified files: {len([f for f in status['files'] if f['status'] == 'M'])}")
print(f"  - Untracked files: {len([f for f in status['files'] if f['status'] == '??'])}")

# 2. تاریخچه اخیر
log = registry.git.git_log(".", max_count=5)
print(f"\n📜 Recent Commits:")
for commit in log['commits']:
    print(f"  - {commit['message']} by {commit['author']}")

# 3. لیست فایل‌های Python
files = registry.filesystem.list_directory(".", pattern="*.py", recursive=True)
print(f"\n🐍 Python Files: {files['count']}")

# 4. جستجوی TODO در کد
todos = registry.filesystem.search_files(".", pattern="*.py", content_pattern="TODO")
print(f"\n✅ TODOs found in {todos['count']} files")
```

### مثال 2: تولید گزارش پروژه

```python
from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry
from pathlib import Path

registry = MDSkillsMCPRegistry()

# جمع‌آوری اطلاعات
git_log = registry.git.git_log(".", max_count=10)
py_files = registry.filesystem.list_directory(".", pattern="*.py", recursive=True)
md_files = registry.filesystem.list_directory(".", pattern="*.md", recursive=True)

# تولید گزارش
report = f"""
# Project Report

## Git Statistics
- Total commits (last 10): {len(git_log['commits'])}
- Last commit: {git_log['commits'][0]['message']}

## Code Statistics
- Python files: {py_files['count']}
- Documentation files: {md_files['count']}

## Recent Activity
"""

for commit in git_log['commits'][:5]:
    report += f"- {commit['date']}: {commit['message']}\n"

# ذخیره گزارش
registry.filesystem.write_file("PROJECT_REPORT.md", report)
print("✅ Report generated: PROJECT_REPORT.md")
```

### مثال 3: جستجوی GitHub و تحلیل

```python
import asyncio
from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry

async def analyze_github_repos(query: str):
    registry = MDSkillsMCPRegistry()
    
    # جستجوی repositories
    search_result = await registry.github.search_repositories(query, max_results=5)
    
    print(f"🔍 Search Results for '{query}':")
    print(f"Total found: {search_result['total']}\n")
    
    # تحلیل هر repository
    for repo in search_result['repositories']:
        print(f"📦 {repo['full_name']}")
        print(f"   ⭐ Stars: {repo['stars']}")
        print(f"   🍴 Forks: {repo['forks']}")
        print(f"   💻 Language: {repo['language']}")
        print(f"   📝 {repo['description']}\n")
        
        # دریافت اطلاعات تکمیلی
        owner, name = repo['full_name'].split('/')
        info = await registry.github.get_repository_info(owner, name)
        
        if info['success']:
            print(f"   🏷️  Topics: {', '.join(info['topics'])}")
            print(f"   📅 Created: {info['created_at']}")
            print(f"   🔄 Updated: {info['updated_at']}\n")

# اجرا
asyncio.run(analyze_github_repos("agentic rag"))
```

### مثال 4: تولید مستندات خودکار

```python
from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry
from pathlib import Path

registry = MDSkillsMCPRegistry()

# پیدا کردن تمام ماژول‌های Python
modules = registry.filesystem.list_directory("src/", pattern="*.py", recursive=True)

print(f"📚 Generating documentation for {modules['count']} modules...")

# تولید مستندات برای هر ماژول
for module in modules['items']:
    if module['is_dir']:
        continue
    
    module_path = module['path']
    module_name = Path(module_path).stem
    output_path = f"docs/api/{module_name}.md"
    
    result = registry.docs.generate_api_docs(
        code_path=module_path,
        output_path=output_path,
        format="markdown"
    )
    
    if result['success']:
        print(f"✅ {module_name}: {result['items_documented']} items documented")
    else:
        print(f"❌ {module_name}: {result['error']}")

print("\n🎉 Documentation generation complete!")
```

---

## یکپارچگی با MCP Server

### استفاده در Claude Desktop

1. **نصب MCP Server**:
```bash
cd /Users/dbk/Desktop/agentic-graph-RAG
python3 mcp/mcp_server.py
```

2. **پیکربندی Claude Desktop**:
```json
{
  "mcpServers": {
    "agentic-graph-rag": {
      "command": "python3",
      "args": ["/Users/dbk/Desktop/agentic-graph-RAG/mcp/mcp_server.py"]
    }
  }
}
```

3. **استفاده در Claude**:
```
User: Check git status of the project
Claude: [Uses git_status tool]

User: Search for Python repositories about RAG
Claude: [Uses github_search_repos tool]

User: Read the README file
Claude: [Uses read_file tool]
```

### Tools موجود در MCP Server

تمام 15 tool در MCP server یکپارچه شده‌اند:

```python
# Git Tools
- git_status
- git_log
- git_diff

# GitHub Tools
- github_search_repos
- github_repo_info

# File System Tools
- read_file
- write_file
- list_directory
- search_files

# Database Tools
- query_sqlite
- list_sqlite_tables

# Web Tools
- fetch_url
- scrape_webpage

# Documentation Tools
- generate_api_docs
```

---

## تست سیستم

برای تست کامل:

```bash
# تست MDSkills MCP tools
python3 tests/test_mdskills_mcp.py

# تست MCP server
python3 mcp/mdskills_mcp_tools.py
```

خروجی مورد انتظار:
```
✅ PASSED: Git Tools
✅ PASSED: File System Tools
✅ PASSED: Database Tools
✅ PASSED: Documentation Tools
✅ PASSED: Tool Registry

Total: 5/5 tests passed
🎉 All tests passed!
```

---

## نکات مهم

1. **GitHub Token**: برای استفاده از GitHub tools، token را تنظیم کنید:
```bash
export GITHUB_TOKEN="your_token_here"
```

2. **Async Tools**: برخی tools async هستند:
```python
# ❌ اشتباه
result = registry.github.search_repositories("query")

# ✅ درست
result = await registry.github.search_repositories("query")
```

3. **Error Handling**: همیشه `success` را بررسی کنید:
```python
result = registry.git.git_status(".")
if result["success"]:
    print(f"Files: {len(result['files'])}")
else:
    print(f"Error: {result['error']}")
```

4. **File Paths**: مسیرهای نسبی به current directory هستند:
```python
# نسبی
registry.filesystem.read_file("README.md")

# مطلق
registry.filesystem.read_file("/full/path/to/file.txt")
```

---

## منابع بیشتر

- 📚 [MDSkills.ai Documentation](https://www.mdskills.ai/)
- 🔗 [MCP Protocol Specification](https://modelcontextprotocol.io/)
- 🐙 [GitHub MCP Servers](https://github.com/modelcontextprotocol)
- 💬 [Claude Desktop MCP Guide](https://docs.anthropic.com/claude/docs/mcp)

---

**آخرین بروزرسانی**: 2026-06-07  
**نسخه**: 1.0.0  
**تعداد Tools**: 15  
**وضعیت**: ✅ Production Ready