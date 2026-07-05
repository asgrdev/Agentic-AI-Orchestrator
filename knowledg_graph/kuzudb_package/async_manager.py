from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from .client import KuzuClient, QueryResult
from .manager import GraphEdge, GraphNode, KuzuManager 

logger = logging.getLogger(__name__)


class AsyncKuzuManager:
    """
    Wrapper غیرهمزمان برای KuzuManager.

    تغییرات نسبت به نسخه قبل:

    1. رفع asyncio.get_event_loop() deprecated:
       از asyncio.get_running_loop() در هر متد استفاده می‌شود.

    2. Thread Safety:
       هر thread در pool از ConnectionPool یک connection مستقل می‌گیرد.

    3. پوشش کامل متدهای KuzuManager:
       stats, health_check, find_path, get_chunks_by_entity اضافه شد.
    """

    def __init__(
        self,
        db_path: str,
        max_workers: int = 4,
    ) -> None:
        self._client = KuzuClient(db_path)
        self._manager: Optional[KuzuManager] = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="kuzu-worker",
        )

    async def __aenter__(self) -> "AsyncKuzuManager":
        # idempotent — نمونه مشترک بین چند agent ممکن است دوبار startup شود
        if self._manager is not None:
            return self
        await self._run(self._client.connect)
        self._manager = KuzuManager(self._client)
        await self._run(self._manager.setup_schema)
        return self

    async def __aexit__(self, *_) -> None:
        if self._manager is None:
            return
        await self._run(self._client.disconnect)
        self._executor.shutdown(wait=True)
        self._manager = None

    # ── اجرای sync در thread pool ─────────────

    async def _run(self, fn, *args, **kwargs) -> Any:
        """
        رفع مشکل: asyncio.get_event_loop() deprecated.
        از get_running_loop() استفاده می‌شود که فقط در context async معتبر است.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: fn(*args, **kwargs),
        )

    # ── Entity ────────────────────────────────

    async def upsert_entity(self, node: GraphNode) -> QueryResult:
        return await self._run(self._manager.upsert_entity, node)

    async def upsert_relation(self, edge: GraphEdge) -> QueryResult:
        return await self._run(self._manager.upsert_relation, edge)

    async def batch_upsert(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> dict[str, int]:
        return await self._run(self._manager.batch_upsert, nodes, edges)

    # ── Document & Chunk ──────────────────────

    async def insert_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        source_url: str = "",
    ) -> QueryResult:
        return await self._run(
            self._manager.insert_document,
            doc_id, title, content, source_url,
        )

    async def insert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        chunk_index: int,
        embedding: Optional[list[float]] = None,
    ) -> None:
        return await self._run(
            self._manager.insert_chunk,
            chunk_id, doc_id, content, chunk_index, embedding,
        )

    async def link_chunk_to_entity(
        self,
        chunk_id: str,
        entity_id: str,
        confidence: float = 1.0,
    ) -> QueryResult:
        return await self._run(
            self._manager.link_chunk_to_entity,
            chunk_id, entity_id, confidence,
        )

    # ── Query ─────────────────────────────────

    async def find_entity_by_name(self, name: str, limit: int = 10) -> QueryResult:
        return await self._run(self._manager.find_entity_by_name, name, limit)

    async def get_entity_context(self, entity_id: str) -> Optional[dict[str, Any]]:
        return await self._run(self._manager.get_entity_context, entity_id)

    async def get_neighbors(self, entity_id: str, hops: int = 1) -> QueryResult:
        return await self._run(self._manager.get_neighbors, entity_id, hops)

    async def find_path(self, entity_a_id: str, entity_b_id: str) -> QueryResult:
        return await self._run(self._manager.find_path, entity_a_id, entity_b_id)

    async def get_chunks_by_entity(self, entity_id: str) -> QueryResult:
        return await self._run(self._manager.get_chunks_by_entity, entity_id)

    # ── Utility ───────────────────────────────

    async def stats(self) -> dict[str, int]:
        return await self._run(self._manager.stats)

    async def health_check(self) -> bool:
        return await self._run(self._manager.health_check)

    async def setup_schema(self) -> None:
        return await self._run(self._manager.setup_schema)

    async def execute(self, query: str, parameters: Optional[dict] = None) -> QueryResult:
        """Execute a raw Cypher query"""
        return await self._run(self._client.execute, query, parameters or {})
