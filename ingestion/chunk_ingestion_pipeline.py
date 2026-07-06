# ingestion/chunk_ingestion_pipeline.py
"""
ChunkIngestionPipeline

وظیفه:
    CollectedItem / raw text → chunk → embed → entity/relation extraction
    → upsert Weaviate + KuzuDB (nodes + MENTIONS edges)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
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
        embed_concurrency: int = 4,
    ):
        self._weaviate = weaviate
        self._kuzu = kuzu
        self._embed_fn = embed_fn
        self._entity_pipeline = entity_pipeline
        self._relation_pipeline = relation_pipeline
        self._graph_builder = GraphBuilder(kuzu) if kuzu else None
        self._chunk_size = chunk_size
        # قبلاً برای هر chunk یک thread هم‌زمان ساخته می‌شد (gather بدون سقف)
        self._embed_sem = asyncio.Semaphore(max(1, embed_concurrency))

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
            item_chunks = DataHelper.chunk_text(item.content, self._chunk_size)
            logger.debug(
                "✂️ chunked '%s' | %d chars → %d chunks",
                (item.title or item.id)[:60], len(item.content or ""), len(item_chunks),
            )
            for i, text in enumerate(item_chunks):
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
        if items:
            total_chars = sum(len(item.content or "") for item in items)
            logger.info(
                "✂️ Chunking done | items=%d | chunks=%d | total=%d chars | "
                "avg=%d chars/chunk (chunk_size=%d)",
                len(items), len(chunks), total_chars,
                total_chars // max(1, len(chunks)), self._chunk_size,
            )
        return chunks

    async def _ingest_chunks(self, chunks: list[dict]) -> None:
        t_start = time.perf_counter()
        logger.info("📦 Ingest pipeline started | chunks=%d", len(chunks))

        # ── 1. Entity extraction ───────────────────────────────────────
        t0 = time.perf_counter()
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
        logger.info(
            "🏷️ Entity extraction | chunks=%d → entities=%d | %.1fs",
            len(chunks), len(all_entities), time.perf_counter() - t0,
        )

        # ── 2. Relation extraction ─────────────────────────────────────
        t0 = time.perf_counter()
        chunk_relations: dict[str, list] = {}
        relation_failures = 0
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
                    relation_failures += 1
                    logger.warning("Relation extraction failed for chunk %s: %s", chunk["id"], ex)
            total_relations = sum(len(r) for r in chunk_relations.values())
            logger.info(
                "🔗 Relation extraction | relations=%d in %d chunks | failed=%d | %.1fs",
                total_relations, len(chunk_relations), relation_failures,
                time.perf_counter() - t0,
            )

        # ── 3. Embed ───────────────────────────────────────────────────
        embedded_ok = await self._embed_chunks(chunks)

        # ── 4. Weaviate upsert ─────────────────────────────────────────
        # فقط فیلدهایی که امضای WeaviateStore.upsert می‌پذیرد پاس می‌شوند —
        # پاس‌دادن مستقیم chunk (با کلیدهای id/text/title/metadata) باعث
        # «unexpected keyword argument 'id'» می‌شد و هیچ ingest ای کامل نمی‌شد
        t0 = time.perf_counter()
        upsert_items = []
        for idx, c in enumerate(chunks):
            if not c.get("embedding"):
                logger.warning("Chunk %s بدون embedding — از upsert صرف‌نظر شد", c["id"])
                continue
            meta = c.get("metadata") or {}
            upsert_items.append({
                "chunk_id":       c["chunk_id"],
                "content":        c["content"],
                "embedding":      c["embedding"],
                "source":         str(meta.get("source", "")),
                "doc_id":         str(meta.get("doc_id") or c.get("title") or ""),
                "entity_ids":     c.get("entity_ids", []),
                "chunk_index":    idx,
                "extra_metadata": json.dumps(meta, ensure_ascii=False, default=str),
            })
        if upsert_items:
            await self._weaviate.upsert_batch(items=upsert_items)
        logger.info(
            "🗄️ Weaviate upsert | chunks=%d (embedded=%d, upserted=%d) | %.1fs",
            len(chunks), embedded_ok, len(upsert_items), time.perf_counter() - t0,
        )

        # ── 5. KuzuDB: Chunk nodes + Entity nodes + MENTIONS edges ─────
        t0 = time.perf_counter()
        await self._upsert_graph(chunks, all_entities, chunk_relations)
        logger.info("🕸️ Graph upsert | %.1fs", time.perf_counter() - t0)

        logger.info(
            "✅ Ingest pipeline done | %d chunks | %d entities | %d relation-sets "
            "→ Weaviate + KuzuDB | total %.1fs",
            len(chunks), len(all_entities), len(chunk_relations),
            time.perf_counter() - t_start,
        )

    async def _embed_chunks(self, chunks: list[dict]) -> int:
        """برمی‌گرداند: تعداد chunk هایی که embedding موفق گرفتند"""
        t0 = time.perf_counter()
        done = 0

        async def _one(chunk: dict) -> bool:
            nonlocal done
            async with self._embed_sem:
                try:
                    chunk["embedding"] = await asyncio.to_thread(
                        self._embed_fn, chunk["content"]
                    )
                except Exception as e:
                    logger.warning("Embed failed for chunk %s: %s", chunk["id"], e)
                    chunk["embedding"] = None
                    return False
            done += 1
            if done % 20 == 0:
                logger.info("🔢 Embedding progress | %d/%d chunks", done, len(chunks))
            return True

        results = await asyncio.gather(*[_one(c) for c in chunks])
        ok = sum(results)
        failed = len(chunks) - ok
        dim = 0
        for c in chunks:
            if c.get("embedding"):
                dim = len(c["embedding"])
                break
        logger.info(
            "🔢 Embedding stage | chunks=%d | ok=%d | failed=%d | dim=%d | %.1fs",
            len(chunks), ok, failed, dim, time.perf_counter() - t0,
        )
        return ok

    async def _upsert_graph(
        self,
        chunks: list[dict],
        entities: list,
        chunk_relations: dict[str, list],
    ) -> None:
        """Chunk nodes را در KuzuDB ثبت می‌کند، سپس GraphBuilder برای entities/relations."""

        # Chunk nodes — GraphNode واقعی از models (AsyncKuzuManager.GraphNode وجود ندارد)
        from knowledg_graph.kuzudb_package.models import GraphNode

        nodes = [
            GraphNode(
                id=c["id"],
                name=c.get("title") or c["id"],
                canonical_name=c.get("title") or c["id"],
                type="Document",
                description=(c.get("content") or "")[:500],
                kb_url=c["metadata"].get("url", "") or "",
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
