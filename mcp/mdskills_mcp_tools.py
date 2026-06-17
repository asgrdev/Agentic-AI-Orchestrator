"""
MDSkills.ai MCP Tools Integration
==================================

بر اساس https://www.mdskills.ai/mcp-servers و https://www.mdskills.ai/use-cases/developer-tools
این ماژول MCP tools مفید را برای سیستم فراهم می‌کند.

Categories:
1. Developer Tools (Git, GitHub, Code Analysis)
2. File System Operations
3. Database Tools
4. Web & API Tools
5. AI & ML Tools
6. Documentation Tools
"""

import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. DEVELOPER TOOLS
# ═══════════════════════════════════════════════════════════════

class GitMCPTools:
    """
    Git operations via MCP
    Based on: @modelcontextprotocol/server-git
    """
    
    @staticmethod
    def git_status(repo_path: str = ".") -> Dict[str, Any]:
        """Get git repository status"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    status = line[:2].strip()
                    filepath = line[3:]
                    files.append({"status": status, "file": filepath})
            
            return {
                "success": True,
                "repo_path": repo_path,
                "files": files,
                "clean": len(files) == 0
            }
        except Exception as e:
            logger.error(f"Git status failed: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def git_log(repo_path: str = ".", max_count: int = 10) -> Dict[str, Any]:
        """Get git commit history"""
        try:
            result = subprocess.run(
                ["git", "log", f"--max-count={max_count}", "--pretty=format:%H|%an|%ae|%ad|%s"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('|')
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "message": parts[4] if len(parts) > 4 else ""
                    })
            
            return {
                "success": True,
                "repo_path": repo_path,
                "commits": commits
            }
        except Exception as e:
            logger.error(f"Git log failed: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def git_diff(repo_path: str = ".", file_path: Optional[str] = None) -> Dict[str, Any]:
        """Get git diff"""
        try:
            cmd = ["git", "diff"]
            if file_path:
                cmd.append(file_path)
            
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "success": True,
                "repo_path": repo_path,
                "file_path": file_path,
                "diff": result.stdout
            }
        except Exception as e:
            logger.error(f"Git diff failed: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def git_commit(repo_path: str, message: str, files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a git commit"""
        try:
            # Add files
            if files:
                for file in files:
                    subprocess.run(
                        ["git", "add", file],
                        cwd=repo_path,
                        check=True
                    )
            else:
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=repo_path,
                    check=True
                )
            
            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            return {
                "success": True,
                "repo_path": repo_path,
                "message": message,
                "output": result.stdout
            }
        except Exception as e:
            logger.error(f"Git commit failed: {e}")
            return {"success": False, "error": str(e)}


class GitHubMCPTools:
    """
    GitHub operations via MCP
    Based on: @modelcontextprotocol/server-github
    """
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or self._get_token_from_env()
    
    def _get_token_from_env(self) -> Optional[str]:
        """Get GitHub token from environment"""
        import os
        return os.getenv("GITHUB_TOKEN")
    
    async def search_repositories(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search GitHub repositories"""
        try:
            import aiohttp
            
            url = "https://api.github.com/search/repositories"
            params = {"q": query, "per_page": max_results}
            headers = {}
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    data = await response.json()
                    
                    repos = []
                    for item in data.get("items", []):
                        repos.append({
                            "name": item["name"],
                            "full_name": item["full_name"],
                            "description": item.get("description"),
                            "stars": item["stargazers_count"],
                            "forks": item["forks_count"],
                            "language": item.get("language"),
                            "url": item["html_url"]
                        })
                    
                    return {
                        "success": True,
                        "query": query,
                        "total": data.get("total_count", 0),
                        "repositories": repos
                    }
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get repository information"""
        try:
            import aiohttp
            
            url = f"https://api.github.com/repos/{owner}/{repo}"
            headers = {}
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    data = await response.json()
                    
                    return {
                        "success": True,
                        "name": data["name"],
                        "full_name": data["full_name"],
                        "description": data.get("description"),
                        "stars": data["stargazers_count"],
                        "forks": data["forks_count"],
                        "language": data.get("language"),
                        "topics": data.get("topics", []),
                        "created_at": data["created_at"],
                        "updated_at": data["updated_at"],
                        "url": data["html_url"]
                    }
        except Exception as e:
            logger.error(f"GitHub repo info failed: {e}")
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# 2. FILE SYSTEM OPERATIONS
# ═══════════════════════════════════════════════════════════════

class FileSystemMCPTools:
    """
    File system operations via MCP
    Based on: @modelcontextprotocol/server-filesystem
    """
    
    @staticmethod
    def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read file content"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"success": False, "error": "File not found"}
            
            content = path.read_text(encoding=encoding)
            
            return {
                "success": True,
                "file_path": str(path),
                "content": content,
                "size": path.stat().st_size,
                "lines": len(content.split('\n'))
            }
        except Exception as e:
            logger.error(f"Read file failed: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def write_file(file_path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Write content to file"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding=encoding)
            
            return {
                "success": True,
                "file_path": str(path),
                "size": path.stat().st_size
            }
        except Exception as e:
            logger.error(f"Write file failed: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def list_directory(dir_path: str, recursive: bool = False, pattern: str = "*") -> Dict[str, Any]:
        """List directory contents"""
        try:
            path = Path(dir_path)
            if not path.exists():
                return {"success": False, "error": "Directory not found"}
            
            if recursive:
                files = list(path.rglob(pattern))
            else:
                files = list(path.glob(pattern))
            
            items = []
            for file in files:
                items.append({
                    "name": file.name,
                    "path": str(file),
                    "is_dir": file.is_dir(),
                    "size": file.stat().st_size if file.is_file() else 0
                })
            
            return {
                "success": True,
                "directory": str(path),
                "items": items,
                "count": len(items)
            }
        except Exception as e:
            logger.error(f"List directory failed: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def search_files(dir_path: str, pattern: str, content_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Search files by name and optionally content"""
        try:
            path = Path(dir_path)
            if not path.exists():
                return {"success": False, "error": "Directory not found"}
            
            files = list(path.rglob(pattern))
            
            results = []
            for file in files:
                if file.is_file():
                    match_info = {
                        "path": str(file),
                        "name": file.name,
                        "size": file.stat().st_size
                    }
                    
                    # Search content if pattern provided
                    if content_pattern:
                        try:
                            content = file.read_text()
                            if content_pattern in content:
                                match_info["content_match"] = True
                                # Find line numbers
                                lines = content.split('\n')
                                match_lines = [i+1 for i, line in enumerate(lines) if content_pattern in line]
                                match_info["match_lines"] = match_lines[:10]  # First 10 matches
                            else:
                                continue  # Skip if content doesn't match
                        except:
                            pass
                    
                    results.append(match_info)
            
            return {
                "success": True,
                "directory": str(path),
                "pattern": pattern,
                "content_pattern": content_pattern,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Search files failed: {e}")
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# 3. DATABASE TOOLS
# ═══════════════════════════════════════════════════════════════

class DatabaseMCPTools:
    """
    Database operations via MCP
    Based on: @modelcontextprotocol/server-postgres, @modelcontextprotocol/server-sqlite
    """
    
    @staticmethod
    def query_sqlite(db_path: str, query: str) -> Dict[str, Any]:
        """Execute SQLite query"""
        try:
            import sqlite3
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            
            if query.strip().upper().startswith("SELECT"):
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
                
                return {
                    "success": True,
                    "db_path": db_path,
                    "query": query,
                    "columns": columns,
                    "rows": results,
                    "count": len(results)
                }
            else:
                conn.commit()
                return {
                    "success": True,
                    "db_path": db_path,
                    "query": query,
                    "affected_rows": cursor.rowcount
                }
        except Exception as e:
            logger.error(f"SQLite query failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            if 'conn' in locals():
                conn.close()
    
    @staticmethod
    def list_sqlite_tables(db_path: str) -> Dict[str, Any]:
        """List all tables in SQLite database"""
        try:
            import sqlite3
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            return {
                "success": True,
                "db_path": db_path,
                "tables": tables,
                "count": len(tables)
            }
        except Exception as e:
            logger.error(f"List tables failed: {e}")
            return {"success": False, "error": str(e)}
        finally:
            if 'conn' in locals():
                conn.close()


# ═══════════════════════════════════════════════════════════════
# 4. WEB & API TOOLS
# ═══════════════════════════════════════════════════════════════

class WebMCPTools:
    """
    Web and API operations via MCP
    Based on: @modelcontextprotocol/server-fetch
    """
    
    @staticmethod
    async def fetch_url(url: str, method: str = "GET", headers: Optional[Dict] = None, 
                       data: Optional[Dict] = None) -> Dict[str, Any]:
        """Fetch URL content"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, json=data) as response:
                    content = await response.text()
                    
                    return {
                        "success": True,
                        "url": url,
                        "status": response.status,
                        "headers": dict(response.headers),
                        "content": content,
                        "size": len(content)
                    }
        except Exception as e:
            logger.error(f"Fetch URL failed: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def scrape_webpage(url: str, selector: Optional[str] = None) -> Dict[str, Any]:
        """Scrape webpage content"""
        try:
            import aiohttp
            from bs4 import BeautifulSoup
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    if selector:
                        elements = soup.select(selector)
                        content = [elem.get_text(strip=True) for elem in elements]
                    else:
                        content = soup.get_text(strip=True)
                    
                    return {
                        "success": True,
                        "url": url,
                        "selector": selector,
                        "content": content,
                        "title": soup.title.string if soup.title else None
                    }
        except Exception as e:
            logger.error(f"Scrape webpage failed: {e}")
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# 5. DOCUMENTATION TOOLS
# ═══════════════════════════════════════════════════════════════

class DocumentationMCPTools:
    """
    Documentation generation and management tools
    """
    
    @staticmethod
    def generate_api_docs(code_path: str, output_path: str, format: str = "markdown") -> Dict[str, Any]:
        """Generate API documentation from code"""
        try:
            # Simple docstring extraction
            path = Path(code_path)
            if not path.exists():
                return {"success": False, "error": "Code path not found"}
            
            docs = []
            
            if path.is_file():
                files = [path]
            else:
                files = list(path.rglob("*.py"))
            
            for file in files:
                try:
                    content = file.read_text()
                    # Extract docstrings (simple approach)
                    import ast
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            docstring = ast.get_docstring(node)
                            if docstring:
                                docs.append({
                                    "file": str(file),
                                    "name": node.name,
                                    "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                                    "docstring": docstring
                                })
                except:
                    pass
            
            # Generate markdown
            if format == "markdown":
                md_content = "# API Documentation\n\n"
                for doc in docs:
                    md_content += f"## {doc['type'].title()}: `{doc['name']}`\n\n"
                    md_content += f"**File**: `{doc['file']}`\n\n"
                    md_content += f"{doc['docstring']}\n\n---\n\n"
                
                Path(output_path).write_text(md_content)
            
            return {
                "success": True,
                "code_path": str(code_path),
                "output_path": output_path,
                "format": format,
                "items_documented": len(docs)
            }
        except Exception as e:
            logger.error(f"Generate docs failed: {e}")
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# UNIFIED MCP TOOLS REGISTRY
# ═══════════════════════════════════════════════════════════════

class MDSkillsMCPRegistry:
    """
    Unified registry for all MDSkills MCP tools
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.git = GitMCPTools()
        self.github = GitHubMCPTools(token=github_token)
        self.filesystem = FileSystemMCPTools()
        self.database = DatabaseMCPTools()
        self.web = WebMCPTools()
        self.docs = DocumentationMCPTools()
    
    def list_all_tools(self) -> List[Dict[str, str]]:
        """List all available MCP tools"""
        return [
            # Git Tools
            {"category": "Git", "name": "git_status", "description": "Get git repository status"},
            {"category": "Git", "name": "git_log", "description": "Get git commit history"},
            {"category": "Git", "name": "git_diff", "description": "Get git diff"},
            {"category": "Git", "name": "git_commit", "description": "Create a git commit"},
            
            # GitHub Tools
            {"category": "GitHub", "name": "search_repositories", "description": "Search GitHub repositories"},
            {"category": "GitHub", "name": "get_repository_info", "description": "Get repository information"},
            
            # File System Tools
            {"category": "FileSystem", "name": "read_file", "description": "Read file content"},
            {"category": "FileSystem", "name": "write_file", "description": "Write content to file"},
            {"category": "FileSystem", "name": "list_directory", "description": "List directory contents"},
            {"category": "FileSystem", "name": "search_files", "description": "Search files by name and content"},
            
            # Database Tools
            {"category": "Database", "name": "query_sqlite", "description": "Execute SQLite query"},
            {"category": "Database", "name": "list_sqlite_tables", "description": "List SQLite tables"},
            
            # Web Tools
            {"category": "Web", "name": "fetch_url", "description": "Fetch URL content"},
            {"category": "Web", "name": "scrape_webpage", "description": "Scrape webpage content"},
            
            # Documentation Tools
            {"category": "Documentation", "name": "generate_api_docs", "description": "Generate API documentation"},
        ]
    
    async def execute_tool(self, category: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by category and name"""
        try:
            if category == "Git":
                tool = getattr(self.git, tool_name)
                return tool(**kwargs)
            elif category == "GitHub":
                tool = getattr(self.github, tool_name)
                return await tool(**kwargs)
            elif category == "FileSystem":
                tool = getattr(self.filesystem, tool_name)
                return tool(**kwargs)
            elif category == "Database":
                tool = getattr(self.database, tool_name)
                return tool(**kwargs)
            elif category == "Web":
                tool = getattr(self.web, tool_name)
                return await tool(**kwargs)
            elif category == "Documentation":
                tool = getattr(self.docs, tool_name)
                return tool(**kwargs)
            else:
                return {"success": False, "error": f"Unknown category: {category}"}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {"success": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize registry
    registry = MDSkillsMCPRegistry()
    
    # List all tools
    print("Available MCP Tools:")
    print("=" * 60)
    for tool in registry.list_all_tools():
        print(f"[{tool['category']}] {tool['name']}: {tool['description']}")
    
    # Example: Git status
    print("\n" + "=" * 60)
    print("Example: Git Status")
    print("=" * 60)
    result = registry.git.git_status(".")
    print(json.dumps(result, indent=2))
    
    # Example: List files
    print("\n" + "=" * 60)
    print("Example: List Files")
    print("=" * 60)
    result = registry.filesystem.list_directory(".", pattern="*.py")
    print(json.dumps(result, indent=2))

