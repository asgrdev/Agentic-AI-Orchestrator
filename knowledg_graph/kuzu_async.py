"""
Kuzu synchronous است - این wrapper آن را async می‌کند
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .kuzu_client import KuzuGraphStore, GraphNode, GraphEdge


class AsyncKuzuStore:
    """
    Wrapper async برای KuzuGraphStore
    عملیات سنگین را در thread pool اجرا می‌کند
    """

    def __init__(self, config: dict, max_workers: int = 4):
        self._store    = KuzuGraphStore(config)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop     = asyncio.get_event_loop()

    async def _run(self, func, *args, **kwargs):
        """اجرای sync function در thread pool"""
        return await self._loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )

    async def upsert_entity(self, node: GraphNode):
        await self._run(self._store.upsert_entity, node)

    async def upsert_relation(self, edge: GraphEdge):
        await self._run(self._store.upsert_relation, edge)

    async def batch_upsert(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ):
        await self._run(self._store.batch_upsert, nodes, edges)

    async def get_subgraph(
        self,
        entity_ids: list[str],
        max_hops:   int = 2,
    ) -> dict:
        return await self._run(
            self._store.get_subgraph, entity_ids, max_hops
        )

    async def find_entity_by_name(self, name: str) -> list[dict]:
        return await self._run(self._store.find_entity_by_name, name)

    async def find_path(
        self, entity_a: str, entity_b: str
    ) -> list[dict]:
        return await self._run(
            self._store.find_path, entity_a, entity_b
        )

    async def get_entity_context(self, entity_id: str) -> dict:
        return await self._run(
            self._store.get_entity_context, entity_id
        )

    async def get_graph_stats(self) -> dict:
        return await self._run(self._store.get_graph_stats)

    async def execute_raw(
        self, query: str, params: dict = None
    ) -> list[dict]:
        return await self._run(
            self._store.execute_raw, query, params
        )

    def close(self):
        self._store.close()
        self._executor.shutdown(wait=False)
