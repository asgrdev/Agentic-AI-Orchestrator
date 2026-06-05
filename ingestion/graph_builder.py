# ingestion/graph_builder.py
"""
GraphBuilder: ساخت KG از chunk + entities + relations
- KuzuDB سازگار
- async کامل
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from knowledg_graph.kuzudb_package.models import GraphEdge, GraphNode

if TYPE_CHECKING:
    from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
    from datasets.entities import Entity
    from datasets.relations import Relation

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    chunk + entities + relations را به KuzuDB وارد می‌کند.
    """

    def __init__(self, graph_manager: "AsyncKuzuManager") -> None:
        self.graph = graph_manager

    # ── Public API ─────────────────────────────────────────────

    async def build(
        self,
        chunk: dict,
        entities: list["Entity"],
        relations: list["Relation"],
    ) -> None:
        """
        chunk: {"id": ..., "title": ..., "text": ...}
        """
        nodes = self._entities_to_nodes(entities)
        edges = self._relations_to_edges(relations)

        # 1. درج entity nodes
        await self.graph.batch_upsert(nodes, edges)

        # 2. MENTIONS edges (chunk → entity)
        for entity in entities:
            await self._link_chunk_entity(
                chunk_id   = chunk["id"],
                entity_id  = entity.linked_id or entity.id,
                confidence = entity.confidence,
            )

    # ── Private ────────────────────────────────────────────────

    def _entities_to_nodes(
        self, entities: list["Entity"]
    ) -> list[GraphNode]:
        return [
            GraphNode(
                id             = entity.linked_id or entity.id,
                name           = entity.text,
                canonical_name = entity.canonical or entity.text,
                type           = entity.label or entity.type,
                description    = getattr(entity, "description", ""),
                confidence     = entity.confidence,
                linked_id      = entity.linked_id or "",
                kb_url         = getattr(entity, "kb_url", ""),
            )
            for entity in entities
        ]

    def _relations_to_edges(
        self, relations: list["Relation"]
    ) -> list[GraphEdge]:
        return [
            GraphEdge(
                from_id     = r.subject_id or r.subject,
                to_id       = r.object_id  or r.object,
                rel_type    = r.predicate,
                weight      = r.confidence,
                confidence  = r.confidence,
                source_doc  = r.source_doc,
                source_chunk= r.source_chunk,
            )
            for r in relations
            if r.subject_id and r.object_id   # فقط linked entities
        ]

    async def _link_chunk_entity(
        self,
        chunk_id: str,
        entity_id: str,
        confidence: float,
    ) -> None:
        """ایجاد MENTIONS edge بین Chunk و Entity."""
        try:
            await self.graph.execute(
                """
                MATCH (c:Chunk  {id: $cid}),
                      (e:Entity {id: $eid})
                MERGE (c)-[m:MENTIONS]->(e)
                SET m.confidence = $conf
                """,
                parameters={
                    "cid":  chunk_id,
                    "eid":  entity_id,
                    "conf": confidence,
                },
            )
        except Exception as exc:
            logger.warning(
                "MENTIONS edge failed c=%s e=%s: %s",
                chunk_id, entity_id, exc
            )
