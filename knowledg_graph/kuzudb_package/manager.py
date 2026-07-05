
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator, Optional

from .client import KuzuClient, QueryResult
from .models import *
from .helper import (
    ColumnDef,
    KuzuHelper,
    NodeSchema,
 )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Domain Data Classes
# ─────────────────────────────────────────────

@dataclass
class GraphNode:
    id: str
    label: str
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# KuzuManager
# ─────────────────────────────────────────────

class KuzuManager:
    """
    لایه منطق تجاری گراف معنایی (Graph RAG / Knowledge Graph).

    تغییرات نسبت به نسخه قبل:

    1. رفع دسترسی به _conn خصوصی:
       _init_schema از self.client.execute() استفاده می‌کند.

    2. رفع Race Condition در upsert_entity:
       به جای MATCH + CREATE جداگانه، از MERGE اتمی استفاده می‌شود.

    3. رفع asyncio.get_event_loop() deprecated:
       در AsyncKuzuManager رفع شده است.

    4. setup_schema ترتیب درست ایجاد جداول را تضمین می‌کند.
    """

    def __init__(self, client: KuzuClient) -> None:
        self.client = client
        self.helper = KuzuHelper(client)
        self._schema_initialized = False

    # ── Schema ────────────────────────────────

    def setup_schema(self) -> None:
        """
        ترتیب مهم است:
        1. Node table ها اول ساخته می‌شوند.
        2. Rel table ها بعد ساخته می‌شوند (به Node ها وابسته هستند).

        رفع مشکل: _init_schema از self.client.execute() استفاده می‌کند
        به جای self.client._conn.execute() (private access).
        """
        if self._schema_initialized:
            return

        # ── ۱. Node Tables ─────────────────────
        node_schemas = [
            NodeSchema(
                table_name="Entity",
                properties=[
                    ColumnDef("id", "STRING", primary=True),
                    ColumnDef("name", "STRING"),
                    ColumnDef("label", "STRING"),
                    ColumnDef("type", "STRING"),
                    ColumnDef("canonical_name", "STRING"),
                    ColumnDef("description", "STRING"),
                    ColumnDef("source", "STRING"),
                    ColumnDef("kb_url", "STRING"),
                    ColumnDef("confidence", "DOUBLE"),
                    ColumnDef("version", "INT64"),
                    ColumnDef("previous_names", "STRING"),
                    ColumnDef("created_at", "STRING"),
                    ColumnDef("last_updated", "STRING"),
                    ColumnDef("last_seen_chunk", "STRING"),
                    ColumnDef("embedding", "DOUBLE[]"),
                ],
                primary_key="id"
            ),
            NodeSchema(
                table_name="Document",
                properties=[
                    ColumnDef("id", "STRING", primary=True),
                    ColumnDef("title", "STRING"),
                    ColumnDef("content", "STRING"),
                    ColumnDef("source_url", "STRING"),
                    ColumnDef("created_at", "STRING"),
                ],
                primary_key="id"
            ),
            NodeSchema(
                table_name="Chunk",
                properties=[
                    ColumnDef("id", "STRING", primary=True),
                    ColumnDef("content", "STRING"),
                    ColumnDef("chunk_index", "INT64"),
                    ColumnDef("embedding", "DOUBLE[]"),
                ],
                primary_key="id"
            ),
        ]

        for schema in node_schemas:
            self.helper.create_node_table(schema)

        # ── ۲. Rel Tables (بعد از Node ها) ────
        rel_schemas = [
            RelationshipSchema(
                table_name="RELATION",
                from_table="Entity",
                to_table="Entity",
                properties=[
                    ColumnDef("relation_type", "STRING"),
                    ColumnDef("rel_type", "STRING"),
                    ColumnDef("weight", "DOUBLE"),
                    ColumnDef("confidence", "DOUBLE"),
                    ColumnDef("version", "INT64"),
                    ColumnDef("source_doc", "STRING"),
                    ColumnDef("source_chunk", "STRING"),
                    ColumnDef("evidence", "STRING"),
                    ColumnDef("timestamp", "STRING"),
                    ColumnDef("created_at", "STRING"),
                ],
            ),
            RelationshipSchema(
                table_name="MENTIONS",
                from_table="Chunk",
                to_table="Entity",
                properties=[
                    ColumnDef("confidence", "DOUBLE"),
                ],
            ),
            RelationshipSchema(
                table_name="HAS_CHUNK",
                from_table="Document",
                to_table="Chunk",
                properties=[
                    ColumnDef("chunk_index", "INT64"),
                ],
            ),
        ]

        for schema in rel_schemas:
            self.helper.create_rel_table(schema)

        self._schema_initialized = True
        logger.info("Schema کامل راه‌اندازی شد.")

    # ── Entity UPSERT ─────────────────────────

    def upsert_entity(self, node: Any) -> QueryResult:
        """
        رفع Race Condition:
        به جای MATCH + CREATE جداگانه، از MERGE اتمی استفاده می‌شود.

        هر دو شکل GraphNode را می‌پذیرد (duck-typing):
        - manager.GraphNode  (label + properties dict)
        - models.GraphNode   (type/canonical_name/description/kb_url/confidence)
        """
        now = datetime.utcnow().isoformat()
        node_props = getattr(node, "properties", None) or {}
        label = getattr(node, "label", "") or getattr(node, "type", "") or ""
        props = {
            "id": node.id,
            "name": node.name,
            "label": label,
            "type": getattr(node, "type", "") or label,
            "canonical_name": getattr(node, "canonical_name", "") or node.name,
            "description": getattr(node, "description", "") or node_props.get("description", ""),
            "source": node_props.get("source", ""),
            "kb_url": getattr(node, "kb_url", "") or node_props.get("url", ""),
            "confidence": float(getattr(node, "confidence", 1.0) or 1.0),
            "created_at": now,
        }

        # embedding جداگانه handle می‌شود (ممکن است None باشد)
        embedding_part = ""
        if getattr(node, "embedding", None) is not None:
            props["embedding"] = node.embedding
            embedding_part = ", e.embedding = $embedding"

        query = f"""
            MERGE (e:Entity {{id: $id}})
            ON CREATE SET
                e.name = $name,
                e.label = $label,
                e.type = $type,
                e.canonical_name = $canonical_name,
                e.description = $description,
                e.source = $source,
                e.kb_url = $kb_url,
                e.confidence = $confidence,
                e.version = 1,
                e.created_at = $created_at
                {embedding_part}
            ON MATCH SET
                e.name = $name,
                e.label = $label,
                e.type = $type,
                e.description = $description,
                e.last_updated = $created_at
                {embedding_part}
        """
        return self.client.execute(query, props)

    def upsert_relation(self, edge: Any) -> QueryResult:
        """
        MERGE اتمی برای رابطه.
        اگر رابطه وجود داشت weight جمع می‌شود.

        هر دو شکل GraphEdge را می‌پذیرد:
        - manager.GraphEdge (source_id/target_id/relation_type)
        - models.GraphEdge  (from_id/to_id/rel_type + confidence/source_doc/source_chunk)
        """
        now = datetime.utcnow().isoformat()
        source_id = getattr(edge, "source_id", None) or getattr(edge, "from_id", "")
        target_id = getattr(edge, "target_id", None) or getattr(edge, "to_id", "")
        rel_type = getattr(edge, "relation_type", None) or getattr(edge, "rel_type", "RELATED")
        query = """
            MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id})
            MERGE (a)-[r:RELATION {relation_type: $relation_type}]->(b)
            ON CREATE SET
                r.rel_type = $relation_type,
                r.weight = $weight,
                r.confidence = $confidence,
                r.version = 1,
                r.source_doc = $source_doc,
                r.source_chunk = $source_chunk,
                r.timestamp = $created_at,
                r.created_at = $created_at
            ON MATCH SET
                r.weight = r.weight + $weight,
                r.timestamp = $created_at
        """
        return self.client.execute(query, {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": rel_type,
            "weight": float(getattr(edge, "weight", 1.0) or 1.0),
            "confidence": float(getattr(edge, "confidence", 1.0) or 1.0),
            "source_doc": getattr(edge, "source_doc", "") or "",
            "source_chunk": getattr(edge, "source_chunk", "") or "",
            "created_at": now,
        })

    def batch_upsert(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> dict[str, int]:
        """پردازش دسته‌ای node و edge."""
        node_count = 0
        edge_count = 0

        with self.transaction():
            for node in nodes:
                self.upsert_entity(node)
                node_count += 1
            for edge in edges:
                self.upsert_relation(edge)
                edge_count += 1

        logger.info("batch_upsert: %d node, %d edge", node_count, edge_count)
        return {"nodes": node_count, "edges": edge_count}

    # ── Document & Chunk ──────────────────────

    def insert_document(self, doc_id: str, title: str, content: str, source_url: str = "") -> QueryResult:
        now = datetime.utcnow().isoformat()
        query = """
            MERGE (d:Document {id: $id})
            ON CREATE SET
                d.title = $title,
                d.content = $content,
                d.source_url = $source_url,
                d.created_at = $created_at
        """
        return self.client.execute(query, {
            "id": doc_id,
            "title": title,
            "content": content,
            "source_url": source_url,
            "created_at": now,
        })

    def insert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        chunk_index: int,
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Chunk می‌سازد و به Document وصل می‌کند."""
        merge_query = """
            MERGE (c:Chunk {id: $id})
            ON CREATE SET
                c.content = $content,
                c.chunk_index = $chunk_index
        """
        params: dict[str, Any] = {
            "id": chunk_id,
            "content": content,
            "chunk_index": chunk_index,
        }
        if embedding is not None:
            params["embedding"] = embedding
            merge_query = merge_query.rstrip() + ",\n                c.embedding = $embedding"

        self.client.execute(merge_query, params)

        # رابطه HAS_CHUNK
        rel_query = """
            MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id})
            MERGE (d)-[:HAS_CHUNK {chunk_index: $chunk_index}]->(c)
        """
        self.client.execute(rel_query, {
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
        })

    def link_chunk_to_entity(
        self,
        chunk_id: str,
        entity_id: str,
        confidence: float = 1.0,
    ) -> QueryResult:
        query = """
            MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $entity_id})
            MERGE (c)-[:MENTIONS {confidence: $confidence}]->(e)
        """
        return self.client.execute(query, {
            "chunk_id": chunk_id,
            "entity_id": entity_id,
            "confidence": confidence,
        })

    # ── Query ─────────────────────────────────

    def find_entity_by_name(self, name: str, limit: int = 10) -> QueryResult:
        query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $name
            RETURN e
            LIMIT $limit
        """
        return self.client.execute(query, {"name": name, "limit": limit})

    def get_entity_context(self, entity_id: str) -> Optional[dict[str, Any]]:
        """
        رفع مشکل: نتیجه collect() ممکن است null باشد.
        """
        query = """
            MATCH (e:Entity {id: $id})
            OPTIONAL MATCH (e)-[r:RELATION]->(neighbor:Entity)
            RETURN
                e,
                collect(DISTINCT {
                    neighbor_id: neighbor.id,
                    neighbor_name: neighbor.name,
                    relation_type: r.relation_type,
                    weight: r.weight
                }) AS relations
        """
        result = self.client.execute(query, {"id": entity_id})
        first = result.first()
        if not first:
            return None

        # فیلتر کردن null هایی که OPTIONAL MATCH برمی‌گرداند
        relations = [
            r for r in (first.get("relations") or [])
            if r and r.get("neighbor_id") is not None
        ]
        return {
            "entity": first.get("e"),
            "relations": relations,
        }

    def get_neighbors(self, entity_id: str, hops: int = 1) -> QueryResult:
        hops = max(1, min(hops, 5))  # حداکثر ۵ لایه برای جلوگیری از timeout
        query = f"""
            MATCH (e:Entity {{id: $id}})-[:RELATION*1..{hops}]-(neighbor:Entity)
            RETURN DISTINCT neighbor
        """
        return self.client.execute(query, {"id": entity_id})

    def find_path(self, entity_a_id: str, entity_b_id: str) -> QueryResult:
        query = """
            MATCH p =   
                (a:Entity {id: $id_a})-[:RELATION* SHORTEST 1..]->(b:Entity {id: $id_b})
            
            RETURN p
        """
        
        return self.client.execute(query, {"id_a": entity_a_id, "id_b": entity_b_id})

    def get_chunks_by_entity(self, entity_id: str) -> QueryResult:
        query = """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {id: $id})
            RETURN c
            ORDER BY c.chunk_index
        """
        return self.client.execute(query, {"id": entity_id})

    def stats(self) -> dict[str, int]:
        tables = ["Entity", "Document", "Chunk"]
        result = {}
        for table in tables:
            result[table] = self.helper.count_nodes(table)
        return result

    def health_check(self) -> bool:
        try:
            self.client.execute("RETURN 1 AS ok")
            return True
        except Exception:
            return False

    # ── Transaction Context Manager ───────────

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Context manager برای تراکنش."""
        with self.client.transaction():
            yield
