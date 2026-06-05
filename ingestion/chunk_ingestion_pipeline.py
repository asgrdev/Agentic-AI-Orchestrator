# ingestion/chunk_ingestion_pipeline.py
"""
ChunkIngestionPipeline

وظیفه:
    CollectedItem / raw text → chunk → embed → entity/relation extraction
    → upsert Weaviate + KuzuDB (nodes + MENTIONS edges)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from external_sources.data_collector.models import CollectedItem
from external_sources.data_collector.helper import DataHelper
from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
from vector_store.weaviate_client import WeaviateStore
from ingestion.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


class ChunkIngestionPipeline:

    def __init__(
        self,
        weaviate: WeaviateStore,
        kuzu: AsyncKuzuManager,
        embed_fn: Callable[[str], list[float]],
        entity_pipeline=None,    # EntityExtractionPipeline (اختیاری)
        relation_pipeline=None,  # RelationPipeline (اختیاری)
        chunk_size: int = 400,
    ):
        self._weaviate = weaviate
        self._kuzu = kuzu
        self._embed_fn = embed_fn
        self._entity_pipeline = entity_pipeline
        self._relation_pipeline = relation_pipeline
        self._graph_builder = GraphBuilder(kuzu) if kuzu else None
        self._chunk_size = chunk_size

    # ── Public API ─────────────────────────────────────────────────────

    async def ingest_items(self, items: list[CollectedItem]) -> int:
        """
        لیستی از CollectedItem را chunk کرده و ingest می‌کند.
        برمی‌گرداند: تعداد chunk های ingest شده
        """
        chunks = self._build_chunks(items)
        if not chunks:
            return 0
        await self._ingest_chunks(chunks)
        return len(chunks)

    async def ingest_text(self, text: str, source: str = "manual") -> int:
        """
        متن خام را مستقیم ingest می‌کند (برای workflow.ingest_text و MCP).
        """
        raw_chunks = DataHelper.chunk_text(text, self._chunk_size)
        chunks = [
            {
                "id":       DataHelper.make_id(source, f"{i}:{c[:40]}"),  # fix: "id" نه "chunk_id"
                "chunk_id": DataHelper.make_id(source, f"{i}:{c[:40]}"),  # backward compat
                "text":     c,   # fix: "text" برای RelationPipeline
                "content":  c,   # برای Weaviate
                "title":    source,
                "metadata": {"source": source},
                "entity_ids": [],
            }
            for i, c in enumerate(raw_chunks)
        ]
        await self._ingest_chunks(chunks)
        return len(chunks)

    # ── Private ────────────────────────────────────────────────────────

    def _build_chunks(self, items: list[CollectedItem]) -> list[dict]:
        chunks = []
        for item in items:
            for i, text in enumerate(DataHelper.chunk_text(item.content, self._chunk_size)):
                chunk_id = DataHelper.make_id(item.source, f"{item.id}_{i}")
                chunks.append({
                    "id":       chunk_id,   # fix: EntityExtractionPipeline از "id" استفاده می‌کنه
                    "chunk_id": chunk_id,   # backward compat برای Weaviate
                    "text":     text,       # fix: RelationPipeline از "text" استفاده می‌کنه
                    "content":  text,       # برای Weaviate upsert
                    "title":    item.title,
                    "metadata": {
                        "source":       item.source,
                        "domain":       item.domain,
                        "title":        item.title,
                        "url":          item.url,
                        "tags":         item.tags,
                        "published_at": str(item.published_at) if item.published_at else None,
                        **item.metadata,
                    },
                    "entity_ids": [],
                })
        return chunks

    async def _ingest_chunks(self, chunks: list[dict]) -> None:
        # ── 1. Entity extraction ───────────────────────────────────────
        all_entities: list = []
        if self._entity_pipeline:
            try:
                all_entities = await self._entity_pipeline.process_chunks(chunks)
                # map entity_ids به chunk‌ها (با "id" که EntityExtractionPipeline ست می‌کنه)
                for e in all_entities:
                    for chunk in chunks:
                        if chunk["id"] == e.source_chunk:
                            chunk["entity_ids"].append(e.id)
            except Exception as ex:
                logger.warning("Entity extraction failed, continuing: %s", ex)

        # ── 2. Relation extraction ─────────────────────────────────────
        chunk_relations: dict[str, list] = {}
        if self._relation_pipeline and all_entities:
            # گروه‌بندی entities بر اساس chunk
            entity_map: dict[str, list] = {}
            for e in all_entities:
                entity_map.setdefault(e.source_chunk, []).append(e)

            for chunk in chunks:
                chunk_ents = entity_map.get(chunk["id"], [])
                if not chunk_ents:
                    continue
                try:
                    relations = self._relation_pipeline.extract(chunk, chunk_ents)
                    chunk_relations[chunk["id"]] = relations
                except Exception as ex:
                    logger.warning("Relation extraction failed for chunk %s: %s", chunk["id"], ex)

        # ── 3. Embed ───────────────────────────────────────────────────
        await self._embed_chunks(chunks)

        # ── 4. Weaviate upsert ─────────────────────────────────────────
        await self._weaviate.upsert_batch(items=chunks)

        # ── 5. KuzuDB: Chunk nodes + Entity nodes + MENTIONS edges ─────
        await self._upsert_graph(chunks, all_entities, chunk_relations)

        logger.info(
            "Ingested %d chunks | %d entities | %d relation-sets → Weaviate + KuzuDB",
            len(chunks), len(all_entities), len(chunk_relations),
        )

    async def _embed_chunks(self, chunks: list[dict]) -> None:
        async def _one(chunk: dict) -> None:
            try:
                chunk["embedding"] = await asyncio.to_thread(
                    self._embed_fn, chunk["content"]
                )
            except Exception as e:
                logger.warning("Embed failed for chunk %s: %s", chunk["id"], e)
                chunk["embedding"] = None

        await asyncio.gather(*[_one(c) for c in chunks])

    async def _upsert_graph(
        self,
        chunks: list[dict],
        entities: list,
        chunk_relations: dict[str, list],
    ) -> None:
        """Chunk nodes را در KuzuDB ثبت می‌کند، سپس GraphBuilder برای entities/relations."""

        # Chunk nodes
        nodes = [
            AsyncKuzuManager.GraphNode(
                id=c["id"],
                name=c.get("title") or c["id"],
                type="Document",
                properties={
                    "content": c["content"],
                    "source":  c["metadata"].get("source", ""),
                    "url":     c["metadata"].get("url", ""),
                },
            )
            for c in chunks
        ]
        await self._kuzu.batch_upsert(nodes=nodes, edges=[])

        # Entity nodes + MENTIONS edges از طریق GraphBuilder
        if self._graph_builder and entities:
            entity_map: dict[str, list] = {}
            for e in entities:
                entity_map.setdefault(e.source_chunk, []).append(e)

            for chunk in chunks:
                chunk_ents = entity_map.get(chunk["id"], [])
                relations  = chunk_relations.get(chunk["id"], [])
                if not chunk_ents:
                    continue
                try:
                    await self._graph_builder.build(
                        chunk={"id": chunk["id"], "title": chunk.get("title", ""), "text": chunk["text"]},
                        entities=chunk_ents,
                        relations=relations,
                    )
                except Exception as ex:
                    logger.warning("GraphBuilder failed for chunk %s: %s", chunk["id"], ex)
