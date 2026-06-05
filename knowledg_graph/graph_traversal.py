# ingestion/graph_traversal.py
"""
GraphTraversal: پیمایش KuzuDB
- sync retrieve (برای pipeline)
- async find_subgraph / find_path (برای search)
- بدون APOC, بدون Neo4j
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
    from ingestion.ner_extractor import NERExtractor
    from ingestion.entity_linker import EntityLinker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Model
# ─────────────────────────────────────────────

@dataclass
class GraphContext:
    entities: list[dict]
    relations: list[dict]
    chunks: list[str]
    context_text: str


# ─────────────────────────────────────────────
# GraphTraversal
# ─────────────────────────────────────────────

class GraphTraversal:
    """
    پیمایش KuzuDB برای Graph RAG.

    - retrieve()              → sync, برای pipeline
    - find_subgraph()         → async, k-hop با KuzuDB
    - find_path_between()     → async, shortest path با KuzuDB
    """

    def __init__(
        self,
        graph_manager: "AsyncKuzuManager",
        ner_extractor: "NERExtractor",
        entity_linker: "EntityLinker",
    ) -> None:
        self.graph  = graph_manager
        self.ner    = ner_extractor
        self.linker = entity_linker

    # ── Sync Pipeline Entry Point ──────────────────────────────

    def retrieve(self, query: str, hops: int = 2) -> GraphContext:
        """
        Pipeline کامل:
        1. NER روی query
        2. Entity linking
        3. K-hop traversal
        4. chunk retrieval
        5. context building
        """
        # NER
        query_entities = self.ner.extract(query)
        if not query_entities:
            return GraphContext([], [], [], "")

        # Linking (async را در sync اجرا می‌کنیم)
        linked = asyncio.get_event_loop().run_until_complete(
            self.linker.link_entities(query_entities)
        )

        entity_ids = [
            e.linked_id or e.id
            for e in linked
            if e.linked_id or e.id
        ]
        if not entity_ids:
            return GraphContext([], [], [], "")

        # Subgraph
        subgraph = asyncio.get_event_loop().run_until_complete(
            self.find_subgraph(entity_ids, max_hops=hops)
        )

        nodes     = subgraph.get("nodes", [])
        edges     = subgraph.get("edges", [])
        unique    = {n["id"]: n for n in nodes}

        # Chunks
        chunks = asyncio.get_event_loop().run_until_complete(
            self._get_chunks_async(list(unique.values()))
        )

        context_text = self._build_context(
            list(unique.values()), edges, chunks
        )

        return GraphContext(
            entities     = list(unique.values()),
            relations    = edges,
            chunks       = chunks,
            context_text = context_text,
        )

    # ── Async Subgraph ─────────────────────────────────────────

    async def find_subgraph(
        self,
        entity_ids: list[str],
        max_hops: int = 3,
    ) -> dict:
        """
        K-hop subgraph با KuzuDB Cypher.
        max_hops=1 → neighbors مستقیم
        max_hops=2 → neighbors of neighbors
        """
        if max_hops == 1:
            query = """
            MATCH (start:Entity)-[r:RELATION]-(neighbor:Entity)
            WHERE start.id IN $entity_ids
            RETURN
                collect(DISTINCT {
                    id: neighbor.id,
                    name: neighbor.name,
                    type: neighbor.type,
                    description: neighbor.description,
                    confidence: neighbor.confidence
                }) AS nodes,
                collect(DISTINCT {
                    from: start.id,
                    to: neighbor.id,
                    type: r.rel_type,
                    weight: r.weight
                }) AS edges
            """
        else:
            # KuzuDB از variable-length path پشتیبانی می‌کند
            query = f"""
            MATCH (start:Entity)-[r:RELATION*1..{max_hops}]-(neighbor:Entity)
            WHERE start.id IN $entity_ids
            RETURN
                collect(DISTINCT {{
                    id: neighbor.id,
                    name: neighbor.name,
                    type: neighbor.type,
                    description: neighbor.description,
                    confidence: neighbor.confidence
                }}) AS nodes,
                collect(DISTINCT {{
                    from: start.id,
                    to: neighbor.id,
                    type: type(r[-1]),
                    weight: r[-1].weight
                }}) AS edges
            """

        try:
            result = await self.graph.execute(
                query,
                parameters={"entity_ids": entity_ids},
            )
            row = result.first()
            if row:
                return {
                    "nodes": row.get("nodes", []),
                    "edges": row.get("edges", []),
                }
        except Exception as exc:
            logger.warning("find_subgraph error: %s", exc)

        return {"nodes": [], "edges": []}

    # ── Async Shortest Path ────────────────────────────────────

    async def find_path_between(
        self,
        entity_a: str,
        entity_b: str,
        max_hops: int = 5,
    ) -> list[dict]:
        """
        کوتاه‌ترین مسیر بین دو entity با KuzuDB.
        KuzuDB syntax: SHORTEST path
        """
        query = f"""
        MATCH (a:Entity {{id: $a}}),
              (b:Entity {{id: $b}}),
              p = (a)-[:RELATION* SHORTEST 1..{max_hops}]-(b)
        RETURN
            [n IN nodes(p) | {{
                id: n.id,
                name: n.name,
                type: n.type
            }}] AS path_nodes,
            [r IN relationships(p) | {{
                type: r.rel_type,
                weight: r.weight
            }}] AS path_rels
        LIMIT 1
        """

        try:
            result = await self.graph.execute(
                query, parameters={"a": entity_a, "b": entity_b}
            )
            row = result.first()
            if not row:
                return []

            nodes = row.get("path_nodes", [])
            rels  = row.get("path_rels",  []) + [None]

            return [
                {"node": n, "rel": r}
                for n, r in zip(nodes, rels)
            ]

        except Exception as exc:
            logger.warning("find_path_between error: %s", exc)
            return []

    # ── Chunk Retrieval ────────────────────────────────────────

    async def _get_chunks_async(
        self, entities: list[dict]
    ) -> list[str]:
        """
        chunk‌هایی که به entities اشاره دارند را
        از طریق MENTIONS edge بازیابی می‌کند.
        """
        chunks: list[str] = []
        for entity in entities:
            eid = entity.get("id", "")
            if not eid:
                continue
            try:
                result = await self.graph.execute(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: $id})
                    RETURN c.content AS content
                    LIMIT 3
                    """,
                    parameters={"id": eid},
                )
                for row in result.rows:
                    content = row.get("content", "")
                    if content:
                        chunks.append(content)
            except Exception as exc:
                logger.debug(
                    "Chunk retrieval failed for entity %s: %s", eid, exc
                )
        return chunks

    # ── Context Builder ────────────────────────────────────────

    def _build_context(
        self,
        entities: list[dict],
        relations: list[dict],
        chunks: list[str],
    ) -> str:
        lines: list[str] = []

        if entities:
            lines.append("## Entities")
            for e in entities:
                lines.append(
                    f"- {e.get('name', '')} [{e.get('type', '')}]"
                    + (f": {e.get('description','')[:100]}"
                       if e.get("description") else "")
                )

        if relations:
            lines.append("\n## Relations")
            for r in relations[:20]:
                lines.append(
                    f"- {r.get('from','')} "
                    f"--[{r.get('type','')}]--> "
                    f"{r.get('to','')}"
                )

        if chunks:
            lines.append("\n## Evidence")
            for c in chunks[:10]:
                lines.append(c.strip())

        return "\n".join(lines)
