"""
MCP (Model Context Protocol) Server:
سیستم را به عنوان یک MCP tool در دسترس
Claude Desktop / Cursor / سایر کلاینت‌ها قرار می‌دهد
"""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json
from core.config import load_config
import logging
from agents.orchestrator import Orchestrator
from workflow.agentic_graph_rag import AgenticGraphRAGWorkflow

logger = logging.getLogger(__name__)

app = Server("agentic-graph-rag")

config = load_config()
workflow = AgenticGraphRAGWorkflow(config)
orchestrator = Orchestrator(config)


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
            
            
        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:

        logger.error(f"Tool {name} failed: {e}", exc_info=True)

        return [
                   ]


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
