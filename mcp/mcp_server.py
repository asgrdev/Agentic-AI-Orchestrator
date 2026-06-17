"""
MCP (Model Context Protocol) Server:
سیستم را به عنوان یک MCP tool در دسترس
Claude Desktop / Cursor / سایر کلاینت‌ها قرار می‌دهد

Enhanced with MDSkills.ai MCP Tools:
- Git operations
- GitHub integration
- File system operations
- Database tools
- Web scraping
- Documentation generation
"""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json
from core.config import load_config
import logging
from agents.orchestrator import Orchestrator
from workflow.agentic_graph_rag import AgenticGraphRAGWorkflow
from mcp.mdskills_mcp_tools import MDSkillsMCPRegistry

logger = logging.getLogger(__name__)

app = Server("agentic-graph-rag")

config = load_config()
workflow = AgenticGraphRAGWorkflow(config)
orchestrator = Orchestrator(config)

# Initialize MDSkills MCP tools
mdskills_registry = MDSkillsMCPRegistry()


# ─────────────────────────────────────
# LIST TOOLS
# ─────────────────────────────────────

@app.list_tools()
async def list_tools():

    return [

        Tool(
            name="query_knowledge_graph",
            description="Query the knowledge graph using the agentic workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "session_id": {"type": "string"}
                },
                "required": ["query", "session_id"],
            },
        ),

        Tool(
            name="collect_news",
            description="Collect political news articles",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_per_feed": {"type": "integer", "default": 5}
                },
            },
        ),

        Tool(
            name="collect_social",
            description="Collect social media data",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 20}
                },
                "required": ["query"],
            },
        ),

        Tool(
            name="collect_financial",
            description="Collect financial market data",
            inputSchema={
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["tickers"],
            },
        ),

        Tool(
            name="ingest_document",
            description="Ingest document into knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source": {"type": "string"}
                },
                "required": ["text"],
            },
        ),

        Tool(
            name="export_json",
            description="Export collected items to JSON file",
            inputSchema={
                "type": "object",
                "properties": {
                    "items": {"type": "array"},
                    "filepath": {"type": "string"}
                },
                "required": ["items", "filepath"],
            },
        )
        ,
        Tool(
            name="explore_entity",
            description="Explore a specific entity in the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "max_hops":    {"type": "integer", "default": 2},
                },
                "required": ["entity_name"],
            },
        ),
        
        # ─────────────────────────────────────
        # MDSkills MCP Tools - Git
        # ─────────────────────────────────────
        Tool(
            name="git_status",
            description="Get git repository status",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "default": "."}
                },
            },
        ),
        Tool(
            name="git_log",
            description="Get git commit history",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "default": "."},
                    "max_count": {"type": "integer", "default": 10}
                },
            },
        ),
        Tool(
            name="git_diff",
            description="Get git diff",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string", "default": "."},
                    "file_path": {"type": "string"}
                },
            },
        ),
        
        # ─────────────────────────────────────
        # MDSkills MCP Tools - GitHub
        # ─────────────────────────────────────
        Tool(
            name="github_search_repos",
            description="Search GitHub repositories",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 10}
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="github_repo_info",
            description="Get GitHub repository information",
            inputSchema={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "repo": {"type": "string"}
                },
                "required": ["owner", "repo"],
            },
        ),
        
        # ─────────────────────────────────────
        # MDSkills MCP Tools - File System
        # ─────────────────────────────────────
        Tool(
            name="read_file",
            description="Read file content",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="write_file",
            description="Write content to file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["file_path", "content"],
            },
        ),
        Tool(
            name="list_directory",
            description="List directory contents",
            inputSchema={
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False},
                    "pattern": {"type": "string", "default": "*"}
                },
                "required": ["dir_path"],
            },
        ),
        Tool(
            name="search_files",
            description="Search files by name and content",
            inputSchema={
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "content_pattern": {"type": "string"}
                },
                "required": ["dir_path", "pattern"],
            },
        ),
        
        # ─────────────────────────────────────
        # MDSkills MCP Tools - Database
        # ─────────────────────────────────────
        Tool(
            name="query_sqlite",
            description="Execute SQLite query",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_path": {"type": "string"},
                    "query": {"type": "string"}
                },
                "required": ["db_path", "query"],
            },
        ),
        Tool(
            name="list_sqlite_tables",
            description="List all tables in SQLite database",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_path": {"type": "string"}
                },
                "required": ["db_path"],
            },
        ),
        
        # ─────────────────────────────────────
        # MDSkills MCP Tools - Web
        # ─────────────────────────────────────
        Tool(
            name="fetch_url",
            description="Fetch URL content",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string", "default": "GET"}
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="scrape_webpage",
            description="Scrape webpage content",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "selector": {"type": "string"}
                },
                "required": ["url"],
            },
        ),
        
        # ─────────────────────────────────────
        # MDSkills MCP Tools - Documentation
        # ─────────────────────────────────────
        Tool(
            name="generate_api_docs",
            description="Generate API documentation from code",
            inputSchema={
                "type": "object",
                "properties": {
                    "code_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "format": {"type": "string", "default": "markdown"}
                },
                "required": ["code_path", "output_path"],
            },
        ),
    ]


# ─────────────────────────────────────
# CALL TOOL
# ─────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict):

    try:

        # --------------------------------
        # Agentic Query
        # --------------------------------
        if name == "query_knowledge_graph":

            state = await orchestrator.run(
                query=arguments["query"],
                session_id=arguments["session_id"]
            )

            result = {
                "answer": state.final_answer,
                "confidence": state.confidence,
                "entities": state.extracted_entities,
                "iterations": state.iteration
            }

            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )
            ]

        # --------------------------------
        # NEWS COLLECTION
        # --------------------------------
        elif name == "collect_news":

            items = workflow.data_manager.collect_political_news(
                query=arguments.get("query"),
                max_per_feed=arguments.get("max_per_feed", 5)
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        [i.model_dump() for i in items],
                        ensure_ascii=False,
                        indent=2
                    )
                )
            ]

        # --------------------------------
        # SOCIAL COLLECTION
        # --------------------------------
        elif name == "collect_social":

            items = workflow.data_manager.collect_social(
                query=arguments["query"],
                max_results=arguments.get("max_results", 20)
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        [i.model_dump() for i in items],
                        ensure_ascii=False,
                        indent=2
                    )
                )
            ]

        # --------------------------------
        # FINANCIAL DATA
        # --------------------------------
        elif name == "collect_financial":

            items = workflow.data_manager.collect_financial(
                tickers=arguments["tickers"]
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        [i.model_dump() for i in items],
                        ensure_ascii=False,
                        indent=2
                    )
                )
            ]

        # --------------------------------
        # INGEST DOCUMENT
        # --------------------------------
        elif name == "ingest_document":

            result = workflow.ingest_text(
                text=arguments["text"],
                source=arguments.get("source")
            )

            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, indent=2)
                )
            ]

        # --------------------------------
        # EXPORT JSON
        # --------------------------------
        elif name == "export_json":

            workflow.data_manager.export_to_json(
                arguments["items"],
                arguments["filepath"]
            )

            return [
                TextContent(
                    type="text",
                    text=f"Saved to {arguments['filepath']}"
                )
            ]
        elif name == "explore_entity":
            result = await rag.explore_entity(
                entity_name=arguments["entity_name"],
                max_hops=arguments.get("max_hops", 2),
            )
            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2),
            )]    
            
            
        # --------------------------------
        # MDSkills MCP Tools - Git
        # --------------------------------
        elif name == "git_status":
            result = mdskills_registry.git.git_status(
                repo_path=arguments.get("repo_path", ".")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "git_log":
            result = mdskills_registry.git.git_log(
                repo_path=arguments.get("repo_path", "."),
                max_count=arguments.get("max_count", 10)
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "git_diff":
            result = mdskills_registry.git.git_diff(
                repo_path=arguments.get("repo_path", "."),
                file_path=arguments.get("file_path")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        # --------------------------------
        # MDSkills MCP Tools - GitHub
        # --------------------------------
        elif name == "github_search_repos":
            result = await mdskills_registry.github.search_repositories(
                query=arguments["query"],
                max_results=arguments.get("max_results", 10)
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "github_repo_info":
            result = await mdskills_registry.github.get_repository_info(
                owner=arguments["owner"],
                repo=arguments["repo"]
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        # --------------------------------
        # MDSkills MCP Tools - File System
        # --------------------------------
        elif name == "read_file":
            result = mdskills_registry.filesystem.read_file(
                file_path=arguments["file_path"],
                encoding=arguments.get("encoding", "utf-8")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "write_file":
            result = mdskills_registry.filesystem.write_file(
                file_path=arguments["file_path"],
                content=arguments["content"],
                encoding=arguments.get("encoding", "utf-8")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "list_directory":
            result = mdskills_registry.filesystem.list_directory(
                dir_path=arguments["dir_path"],
                recursive=arguments.get("recursive", False),
                pattern=arguments.get("pattern", "*")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "search_files":
            result = mdskills_registry.filesystem.search_files(
                dir_path=arguments["dir_path"],
                pattern=arguments["pattern"],
                content_pattern=arguments.get("content_pattern")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        # --------------------------------
        # MDSkills MCP Tools - Database
        # --------------------------------
        elif name == "query_sqlite":
            result = mdskills_registry.database.query_sqlite(
                db_path=arguments["db_path"],
                query=arguments["query"]
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "list_sqlite_tables":
            result = mdskills_registry.database.list_sqlite_tables(
                db_path=arguments["db_path"]
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        # --------------------------------
        # MDSkills MCP Tools - Web
        # --------------------------------
        elif name == "fetch_url":
            result = await mdskills_registry.web.fetch_url(
                url=arguments["url"],
                method=arguments.get("method", "GET")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        elif name == "scrape_webpage":
            result = await mdskills_registry.web.scrape_webpage(
                url=arguments["url"],
                selector=arguments.get("selector")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        # --------------------------------
        # MDSkills MCP Tools - Documentation
        # --------------------------------
        elif name == "generate_api_docs":
            result = mdskills_registry.docs.generate_api_docs(
                code_path=arguments["code_path"],
                output_path=arguments["output_path"],
                format=arguments.get("format", "markdown")
            )
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:

        logger.error(f"Tool {name} failed: {e}", exc_info=True)

        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

#----------------------------old tmp--------------------------------

# @app.list_tools()
# async def list_tools() -> list[Tool]:
#     return [
#         Tool(
#             name="query_knowledge_graph",
#             description=(
#                 "Query the knowledge graph with automatic retrieval, "
#                 "reasoning, and dynamic knowledge refresh from external sources"
#             ),
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "query":      {"type": "string",
#                                    "description": "The question to answer"},
#                     "session_id": {"type": "string",
#                                    "description": "Optional session ID"},
#                 },
#                 "required": ["query"],
#             },
#         ),
#         Tool(
#             name="ingest_document",
#             description="Ingest a new document into the knowledge graph",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "content": {"type": "string"},
#                     "source":  {"type": "string"},
#                 },
#                 "required": ["content", "source"],
#             },
#         ),
#         Tool(
#             name="explore_entity",
#             description="Explore a specific entity in the knowledge graph",
#             inputSchema={
#                 "type": "object",
#                 "properties": {
#                     "entity_name": {"type": "string"},
#                     "max_hops":    {"type": "integer", "default": 2},
#                 },
#                 "required": ["entity_name"],
#             },
#         ),
#     ]


# @app.call_tool()
# async def call_tool(name: str, arguments: dict) -> list[TextContent]:
#     if name == "query_knowledge_graph":
#         result = await rag.run(
#             query=arguments["query"],
#             session_id=arguments.get("session_id"),
#         )
#         return [TextContent(
#             type="text",
#             text=json.dumps(result, ensure_ascii=False, indent=2),
#         )]

#     elif name == "ingest_document":
#         await rag.ingest(
#             content=arguments["content"],
#             source=arguments["source"],
#         )
#         return [TextContent(type="text", text="Document ingested successfully")]

#     elif name == "explore_entity":
#         result = await rag.explore_entity(
#             entity_name=arguments["entity_name"],
#             max_hops=arguments.get("max_hops", 2),
#         )
#         return [TextContent(
#             type="text",
#             text=json.dumps(result, ensure_ascii=False, indent=2),
#         )]

#     raise ValueError(f"Unknown tool: {name}")


# async def main():
#     async with stdio_server() as (read, write):
#         await app.run(read, write, app.create_initialization_options())


# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
