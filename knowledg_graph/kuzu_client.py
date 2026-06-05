import kuzu
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)

 
@dataclass
class GraphNode:

    id: str
    name: str
    canonical_name: str
    type: str

    description: str = ""
    confidence: float = 1.0

    linked_id: str = ""
    kb_url: str = ""
    embedding: list[float] | None = None
    aliases: list[str] | None


@dataclass
class GraphEdge:

    from_id: str
    to_id: str

    rel_type: str

    weight: float = 1.0
    confidence: float = 1.0

    source_doc: str = ""
    source_chunk: str = ""


@dataclass
class GraphChunk:

    id: str
    content: str
    doc_id: str
    chunk_idx: int


@dataclass
class GraphDocument:

    id: str
    title: str
    source: str

class KuzuGraphStore:

    def __init__(self, config: dict):

        db_path = config.get("kuzu_db_path", "./data/kuzu_db")

        Path(db_path).mkdir(parents=True, exist_ok=True)

        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)

        self._create_schema()

        logger.info(f"Kuzu DB initialized at: {db_path}")

    # ------------------------------------------------
    # SCHEMA
    # ------------------------------------------------

    def _create_schema(self):

        schema_queries = [

        """
        CREATE NODE TABLE IF NOT EXISTS Entity(
            id STRING PRIMARY KEY,
            name STRING,
            canonical_name STRING,
            type STRING,
            description STRING,
            confidence DOUBLE,
            linked_id STRING,
            kb_url STRING
        )
        """,

        """
        CREATE NODE TABLE IF NOT EXISTS Document(
            id STRING PRIMARY KEY,
            title STRING,
            source STRING
        )
        """,

        """
        CREATE NODE TABLE IF NOT EXISTS Chunk(
            id STRING PRIMARY KEY,
            content STRING,
            doc_id STRING,
            chunk_idx INT64
        )
        """,

        """
        CREATE REL TABLE IF NOT EXISTS RELATION(
            FROM Entity TO Entity,
            type STRING,
            weight DOUBLE,
            confidence DOUBLE,
            source_doc STRING,
            source_chunk STRING
        )
        """,

        """
        CREATE REL TABLE IF NOT EXISTS MENTIONS(
            FROM Chunk TO Entity,
            confidence DOUBLE
        )
        """,

        """
        CREATE REL TABLE IF NOT EXISTS HAS_CHUNK(
            FROM Document TO Chunk
        )
        """,

        """
        CREATE REL TABLE IF NOT EXISTS DERIVED_FROM(
            FROM Document TO Document,
            weight DOUBLE
        )
        """
        ]

        for q in schema_queries:
            try:
                self._conn.execute(q)
            except Exception:
                pass

    # ------------------------------------------------
    # ENTITY
    # ------------------------------------------------

    def upsert_entity(self, node):

        check = self._conn.execute(
            "MATCH (e:Entity {id:$id}) RETURN e.id",
            parameters={"id": node.id}
        )

        if check.has_next():

            self._conn.execute(
            """
            MATCH (e:Entity {id:$id})
            SET
                e.name=$name,
                e.canonical_name=$canonical,
                e.type=$type,
                e.description=$desc,
                e.confidence=$conf,
                e.linked_id=$linked,
                e.kb_url=$url
            """,
            parameters={
                "id":node.id,
                "name":node.name,
                "canonical":node.canonical_name,
                "type":node.type,
                "desc":node.description,
                "conf":node.confidence,
                "linked":node.linked_id,
                "url":node.kb_url
            }
            )

        else:

            self._conn.execute(
            """
            CREATE (e:Entity {
                id:$id,
                name:$name,
                canonical_name:$canonical,
                type:$type,
                description:$desc,
                confidence:$conf,
                linked_id:$linked,
                kb_url:$url
            })
            """,
            parameters={
                "id":node.id,
                "name":node.name,
                "canonical":node.canonical_name,
                "type":node.type,
                "desc":node.description,
                "conf":node.confidence,
                "linked":node.linked_id,
                "url":node.kb_url
            }
            )

    # ------------------------------------------------
    # RELATIONS
    # ------------------------------------------------

    def upsert_relation(self, edge):

        try:

            self._conn.execute(
            """
            MATCH (a:Entity {id:$from}),
                  (b:Entity {id:$to})

            MERGE (a)-[r:RELATION {type:$type}]->(b)

            ON CREATE
                SET r.weight=$weight,
                    r.confidence=$conf,
                    r.source_doc=$doc,
                    r.source_chunk=$chunk

            ON MATCH
                SET r.weight = r.weight + $weight
            """,
            parameters={
                "from":edge.from_id,
                "to":edge.to_id,
                "type":edge.rel_type,
                "weight":edge.weight,
                "conf":edge.confidence,
                "doc":edge.source_doc,
                "chunk":edge.source_chunk
            }
            )

        except Exception as e:

            logger.error(f"Relation insert error {e}")

    # ------------------------------------------------
    # DOCUMENT
    # ------------------------------------------------

    def insert_document(self, doc):

        self._conn.execute(
        """
        MERGE (d:Document {id:$id})
        SET d.title=$title,
            d.source=$source
        """,
        parameters={
            "id":doc.id,
            "title":doc.title,
            "source":doc.source
        }
        )

    # ------------------------------------------------
    # CHUNK
    # ------------------------------------------------

    def insert_chunk(self, chunk):

        self._conn.execute(
        """
        MERGE (c:Chunk {id:$id})
        SET c.content=$content,
            c.doc_id=$doc,
            c.chunk_idx=$idx
        """,
        parameters={
            "id":chunk.id,
            "content":chunk.content,
            "doc":chunk.doc_id,
            "idx":chunk.chunk_idx
        }
        )

        self._conn.execute(
        """
        MATCH (d:Document {id:$doc}),
              (c:Chunk {id:$chunk})
        MERGE (d)-[:HAS_CHUNK]->(c)
        """,
        parameters={
            "doc":chunk.doc_id,
            "chunk":chunk.id
        }
        )

    # ------------------------------------------------
    # CHUNK → ENTITY
    # ------------------------------------------------

    def add_chunk_entity(self, chunk_id, entity_id, confidence):

        self._conn.execute(
        """
        MATCH (c:Chunk {id:$c}),
              (e:Entity {id:$e})

        MERGE (c)-[r:MENTIONS]->(e)

        SET r.confidence=$conf
        """,
        parameters={
            "c":chunk_id,
            "e":entity_id,
            "conf":confidence
        }
        )

    # ------------------------------------------------
    # BATCH
    # ------------------------------------------------

    def batch_upsert(self, nodes, edges):

        for n in nodes:
            self.upsert_entity(n)

        for e in edges:
            self.upsert_relation(e)

        logger.info(f"Inserted {len(nodes)} nodes / {len(edges)} edges")
  # ─────────────────────────────────────────────
    # READ OPERATIONS - همان Cypher Neo4j
    # ─────────────────────────────────────────────

    def find_entity_by_name(self, name: str) -> list[dict]:
        """جستجو بر اساس نام"""
        result = self._conn.execute(
            """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $name
            RETURN e.id       AS id,
                   e.name     AS name,
                   e.type     AS type,
                   e.description AS description,
                   e.linked_id   AS linked_id
            LIMIT 10
            """,
            parameters={"name": name}
        )
        return self._to_list(result)

    def get_subgraph(
        self,
        entity_ids: list[str],
        max_hops:   int = 2,
    ) -> dict:
        """
        استخراج زیرگراف - معادل Neo4j APOC path
        Kuzu از recursive patterns پشتیبانی می‌کند
        """
        nodes_result = self._conn.execute(
            f"""
            MATCH (start:Entity)
            WHERE start.id IN $entity_ids

            MATCH (start)-[r*1..{max_hops}]-(neighbor:Entity)

            WITH DISTINCT neighbor AS n
            RETURN n.id          AS id,
                   n.name        AS name,
                   n.type        AS type,
                   n.description AS description,
                   n.confidence  AS confidence
            """,
            parameters={"entity_ids": entity_ids}
        )

        edges_result = self._conn.execute(
            f"""
            MATCH (start:Entity)
            WHERE start.id IN $entity_ids

            MATCH (a:Entity)-[r:RELATES_TO|SUPPORTS|CONTRADICTS*1..{max_hops}]-(b:Entity)
            WHERE a.id IN $entity_ids OR b.id IN $entity_ids

            RETURN DISTINCT
                   a.id      AS from_id,
                   b.id      AS to_id,
                   r[0].weight    AS weight,
                   r[0].rel_label AS rel_type
            """,
            parameters={"entity_ids": entity_ids}
        )

        return {
            "nodes": self._to_list(nodes_result),
            "edges": self._to_list(edges_result),
        }
    
    def find_path(
        self,
        entity_a: str,
        entity_b: str,
        max_hops: int = 5,
    ) -> list[dict]:
        """کوتاه‌ترین مسیر بین دو موجودیت"""
    
        result = self._conn.execute(
            f"""
            MATCH (a:Entity {{id: $a}}),
                  (b:Entity {{id: $b}})
            MATCH p = (a)-[:RELATION* SHORTEST 1..{max_hops}]-(b)
            RETURN nodes(p) AS path_nodes,
                   relationships(p) AS path_rels
            LIMIT 1
            """,
            parameters={"a": entity_a, "b": entity_b}
        )
    
        if not result.has_next():
            return []
    
        row = result.get_next()
        nodes = row[0]
        rels = row[1]
    
        return [
            {"node": n, "rel": r}
            for n, r in zip(nodes, rels + [None])
        ]
    
    
    def get_entity_context(self, entity_id: str) -> dict:
        """
        کل context یک موجودیت:
        - خودش
        - همه روابطش
        - موجودیت‌های مرتبط
        """
        result = self._conn.execute(
            """
            MATCH (e:Entity {id: $id})
            OPTIONAL MATCH (e)-[r]-(neighbor:Entity)
            RETURN e.id          AS entity_id,
                   e.name        AS entity_name,
                   e.type        AS entity_type,
                   e.description AS entity_desc,
                   collect({
                       neighbor_id:   neighbor.id,
                       neighbor_name: neighbor.name,
                       rel_type:      type(r),
                       weight:        r.weight
                   }) AS relations
            """,
            parameters={"id": entity_id}
        )

        rows = self._to_list(result)
        return rows[0] if rows else {}

    def search_by_type(
        self,
        entity_type: str,
        limit: int = 20,
    ) -> list[dict]:
        result = self._conn.execute(
            """
            MATCH (e:Entity {type: $type})
            RETURN e.id, e.name, e.description
            LIMIT $limit
            """,
            parameters={"type": entity_type, "limit": limit}
        )
        return self._to_list(result)

    # ─────────────────────────────────────────────
    # ANALYTICS
    # ─────────────────────────────────────────────

    def get_graph_stats(self) -> dict:
        """آمار کلی گراف"""
        entity_count = self._conn.execute(
            "MATCH (e:Entity) RETURN count(e) AS cnt"
        ).get_next()[0]

        rel_count = self._conn.execute(
            "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS cnt"
        ).get_next()[0]

        return {
            "total_entities":  entity_count,
            "total_relations": rel_count,
        }

    def get_most_connected(self, limit: int = 10) -> list[dict]:
        """پرارتباط‌ترین موجودیت‌ها (مثل PageRank ساده)"""
        result = self._conn.execute(
            """
            MATCH (e:Entity)-[r]-()
            RETURN e.id   AS id,
                   e.name AS name,
                   e.type AS type,
                   count(r) AS degree
            ORDER BY degree DESC
            LIMIT $limit
            """,
            parameters={"limit": limit}
        )
        return self._to_list(result)

    # ─────────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────────

    def _to_list(self, query_result) -> list[dict]:
        """تبدیل نتیجه Kuzu به list of dict"""
        rows = []
        col_names = query_result.get_column_names()
        while query_result.has_next():
            row = query_result.get_next()
            rows.append(dict(zip(col_names, row)))
        return rows
    # ------------------------------------------------
    # QUERY
    # ------------------------------------------------

    def get_neighbors(self, entity_id, hops=2):

        result = self._conn.execute(
        f"""
        MATCH (start:Entity {{id:$id}})

        MATCH (start)-[r:RELATION*1..{hops}]-(neighbor)

        RETURN DISTINCT
            neighbor.id as id,
            neighbor.name as name,
            neighbor.type as type
        """,
        parameters={"id":entity_id}
        )

        return self._to_list(result)

    # ------------------------------------------------

    def _to_list(self, result):

        rows=[]

        cols=result.get_column_names()

        while result.has_next():

            row=result.get_next()

            rows.append(dict(zip(cols,row)))

        return rows

    def execute_raw(self, query: str, params: dict = None) -> list[dict]:
        """اجرای مستقیم Cypher - برای انعطاف بیشتر"""
        result = self._conn.execute(query, parameters=params or {})
        return self._to_list(result)

    def close(self):
        self._conn.close()
        self._db.close()
