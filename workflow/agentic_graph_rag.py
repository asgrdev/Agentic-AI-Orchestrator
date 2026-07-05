"""
AgenticGraphRAGWorkflow - Orchestrator اصلی

Pipeline کامل:
    Query
      ↓
    [1] WeaviateStore.graph_rag_search()  → SearchResults + entity_ids
      ↓
    [2] AsyncKuzuManager.get_subgraph()   → nodes + relations (k-hop)
      ↓
    [3] LLMReasoner.reason()              → CitedAnswer
      ↓
    CitedAnswer با پاسخ + citations + confidence
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ingestion.context_builder import ContextBuilder 

 
from typing import Any, Dict, List, Optional

from external_sources.data_collector.manager import DataCollectorManager as DataManager
from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
from vector_store.weaviate_client import ConnectionMode, WeaviateStore
from llm.reasoning.llm_reasoning import CitedAnswer, LLMReasoner
from knowledg_graph.graph_traversal import GraphTraversal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class CollectedItem:
    id: str
    text: str
    source: str
    metadata: Dict[str, Any]

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata,
        }

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkflowConfig:
    """تنظیمات کامل workflow"""

    # Vector Search
    vector_top_k: int = 10
    hybrid_alpha: float = 0.7       # 0=BM25, 1=vector
    mmr_lambda: float = 0.6         # 0=diversity, 1=relevance
    final_top_k: int = 5

    # Graph Traversal
    graph_hops: int = 2             # عمق k-hop
    max_graph_nodes: int = 50
    max_graph_relations: int = 100

    # Reasoning
    max_cot_steps: int = 3
    use_chain_of_thought: bool = True

    # Entity Boosting
    entity_boost_factor: float = 1.5

    # Weaviate collection
    collection_name: str = "Chunks"


# ══════════════════════════════════════════════════════════════════════════════
# Main Workflow
# ══════════════════════════════════════════════════════════════════════════════

class AgenticGraphRAGWorkflow:
    """
    Orchestrator اصلی Graph RAG.

    استفاده:
        workflow = AgenticGraphRAGWorkflow(
            kuzu_path="data/graph.db",
            weaviate_config={"mode": "embedded"},
            llm_client=GraniteClient(config),
            config=WorkflowConfig(),
        )
        await workflow.startup()
        answer = await workflow.search("Who founded OpenAI?")
        await workflow.shutdown()
    """

    def __init__(
        self,
        kuzu_path: str,
        weaviate_config: dict,
        llm_client: Any,
        config: Optional[WorkflowConfig] = None,
    ) -> None:
        self.config = config or WorkflowConfig()

        # Components
        self.kuzu = AsyncKuzuManager(kuzu_path)
        self.weaviate = WeaviateStore(weaviate_config)
        self.ctx_builder = ContextBuilder()

        self.reasoner = LLMReasoner(
            llm_client=llm_client,
            context_builder=self.ctx_builder,
            max_reasoning_steps=self.config.max_cot_steps,
            use_chain_of_thought=self.config.use_chain_of_thought,
        )

        self._ready = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    async def startup(self) -> None:
        """راه‌اندازی تمام components"""
        logger.info("Starting AgenticGraphRAGWorkflow...")

        await self.kuzu.startup()

        # Weaviate connection
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._connect_weaviate)

        self._ready = True
        logger.info("Workflow ready.")

    async def shutdown(self) -> None:
        """بستن تمام اتصالات"""
        await self.kuzu.shutdown()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._disconnect_weaviate)

        self._ready = False
        logger.info("Workflow shut down.")

    def _connect_weaviate(self) -> None:
        self.weaviate.connect()

    def _disconnect_weaviate(self) -> None:
        try:
            self.weaviate.disconnect()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Main Search                                                          #
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        embedding: Optional[list[float]] = None,
        collection: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> CitedAnswer:
        """
        اجرای کامل Graph RAG pipeline.

        Args:
            query:      سوال کاربر
            embedding:  embedding vector از query (اختیاری، اگر None باشد skip vector search)
            collection: نام collection در Weaviate (پیش‌فرض از config)
            filters:    فیلترهای اضافه برای vector search

        Returns:
            CitedAnswer با پاسخ کامل
        """
        if not self._ready:
            raise RuntimeError("Call startup() first")

        cfg = self.config
        collection_name = collection or cfg.collection_name

        logger.info("Graph RAG search: %s", query)

        # ── Step 1: Vector + Graph Search ──────────────────────────────
        search_results = []
        entity_ids: list[str] = []

        if embedding:
            loop = asyncio.get_running_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.weaviate.graph_rag_search(
                    collection_name=collection_name,
                    query_vector=embedding,
                    query_text=query,
                    top_k=cfg.vector_top_k,
                    alpha=cfg.hybrid_alpha,
                    mmr_lambda=cfg.mmr_lambda,
                    final_k=cfg.final_top_k,
                    filters=filters,
                ),
            )

            # استخراج entity_ids از نتایج
            entity_ids = self._extract_entity_ids(search_results)
            logger.debug("Found %d search results, %d entities", len(search_results), len(entity_ids))

        # ── Step 2: Entity Weights از Graph ────────────────────────────
        entity_weights: dict[str, float] = {}
        if entity_ids:
            entity_weights = await self.kuzu.get_entity_weights(entity_ids)

        # ── Step 3: Re-rank با Entity Boost ────────────────────────────
        if search_results and entity_weights:
            loop = asyncio.get_running_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: self.weaviate.entity_boost(
                    results=search_results,
                    entity_weights=entity_weights,
                    boost_factor=cfg.entity_boost_factor,
                ),
            )

        # ── Step 4: Graph Subgraph Retrieval ───────────────────────────
        graph_nodes, graph_relations = await self.kuzu.get_subgraph(
            entity_ids=entity_ids,
            hops=cfg.graph_hops,
            max_nodes=cfg.max_graph_nodes,
        )
        logger.debug(
            "Subgraph: %d nodes, %d relations",
            len(graph_nodes),
            len(graph_relations),
        )

        # ── Step 5: LLM Reasoning ──────────────────────────────────────
        answer = await self.reasoner.reason(
            query=query,
            graph_nodes=graph_nodes,
            graph_relations=graph_relations,
            search_results=search_results,
        )

        logger.info(
            "Answer generated (confidence=%.2f, citations=%d)",
            answer.confidence,
            len(answer.citations),
        )

        return answer

    # ------------------------------------------------------------------ #
    # Ingestion                                                            #
    # ------------------------------------------------------------------ #

    async def ingest_document(
        self,
        document: dict,
        collection: Optional[str] = None,
    ) -> dict:
        """
        Ingest یک document به Weaviate.

        Args:
            document: dict با کلیدهای:
                - chunk_id (str)
                - content (str)
                - embedding (list[float])
                - metadata (dict, اختیاری)
                - entity_ids (list[str], اختیاری)

        Returns:
            {"status": "ok", "chunk_id": ...}
        """
        collection_name = collection or self.config.collection_name
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None,
            lambda: self.weaviate.upsert(
                collection_name=collection_name,
                chunk_id=document["chunk_id"],
                content=document["content"],
                embedding=document["embedding"],
                metadata=document.get("metadata", {}),
                entity_ids=document.get("entity_ids", []),
            ),
        )
        return result

    async def ingest_batch(
        self,
        documents: list[dict],
        collection: Optional[str] = None,
        batch_size: int = 100,
    ) -> dict:
        """
        Ingest دسته‌ای از documents.

        Returns:
            {"inserted": N, "updated": M, "errors": K}
        """
        collection_name = collection or self.config.collection_name
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None,
            lambda: self.weaviate.upsert_batch(
                collection_name=collection_name,
                items=documents,
                batch_size=batch_size,
            ),
        )
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _extract_entity_ids(self, search_results: list[Any]) -> list[str]:
        """استخراج entity_ids از search results"""
        ids: list[str] = []
        seen: set[str] = set()

        for result in search_results:
            # SearchResult dataclass
            if hasattr(result, "metadata"):
                meta = result.metadata
            elif isinstance(result, dict):
                meta = result.get("metadata", {})
            else:
                continue

            entity_ids = meta.get("entity_ids", [])
            if isinstance(entity_ids, str):
                entity_ids = [entity_ids]

            for eid in entity_ids:
                if eid and eid not in seen:
                    ids.append(eid)
                    seen.add(eid)

        return ids


# ══════════════════════════════════════════════════════════════════════════════
# Convenience factory
# ══════════════════════════════════════════════════════════════════════════════

def create_workflow(
    kuzu_path: str,
    weaviate_mode: str = "docker", #"embedded",
    llm_backend: str = "mlx",
    llm_config: Optional[dict] = None,
    workflow_config: Optional[WorkflowConfig] = None,
) -> AgenticGraphRAGWorkflow:
    """
    Factory برای ساخت آسان workflow.

    Args:
        kuzu_path:       مسیر پایگاه داده KuzuDB
        weaviate_mode:   "embedded" | "docker" | "cloud"
        llm_backend:     "mlx" | "granite_remote"
        llm_config:      تنظیمات LLM
        workflow_config: تنظیمات workflow

    مثال:
        workflow = create_workflow(
            kuzu_path="./data/graph.db",
            weaviate_mode="embedded",
            llm_backend="mlx",
        )
    """
    # ساخت LLM client
    if llm_backend == "mlx":
        from reasoning.llm_clients import MlxGraniteJsonExtractor, MlxGraniteConfig  # noqa
        cfg = MlxGraniteConfig(**(llm_config or {}))
        llm = MlxGraniteJsonExtractor(cfg)
    elif llm_backend == "granite_remote":
        from reasoning.llm_clients import GraniteClient  # noqa
        llm = GraniteClient(llm_config or {})
    else:
        raise ValueError(f"Unknown llm_backend: {llm_backend}")

    # ساخت Weaviate config
    weaviate_cfg = {"mode": weaviate_mode, **(llm_config or {}).get("weaviate", {})}

    return AgenticGraphRAGWorkflow(
        kuzu_path=kuzu_path,
        weaviate_config=weaviate_cfg,
        llm_client=llm,
        config=workflow_config,
    )




# # agentic_workflow.py
# """
# AgenticGraphRAGWorkflow - لایه Data / Infrastructure

# تغییر نسبت به نسخه قبل:
# - VectorIndexManager حذف شد
# - WeaviateStore جایگزین شد (با MMR + Entity Boosting)
# """

# from __future__ import annotations

# import logging
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# from external_sources.data_collector.manager import DataCollectorManager as DataManager
# from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
# from vector_store.weaviate_client import ConnectionMode, WeaviateStore

# logger = logging.getLogger(__name__)


# # ─────────────────────────────────────────────
# # Data Models
# # ─────────────────────────────────────────────

# @dataclass
# class CollectedItem:
#     id: str
#     text: str
#     source: str
#     metadata: Dict[str, Any]

#     def model_dump(self) -> dict:
#         return {
#             "id": self.id,
#             "text": self.text,
#             "source": self.source,
#             "metadata": self.metadata,
#         }


# # ─────────────────────────────────────────────
# # AgenticGraphRAGWorkflow
# # ─────────────────────────────────────────────

# class AgenticGraphRAGWorkflow:
#     """
#     لایه Data / Infrastructure سیستم.

#     orchestrator → reasoning
#     workflow     → ingestion + collectors + indexing
#     """

#     def __init__(self, config: dict) -> None:
#         self.config = config

#         self.data_manager = DataManager()

#         self.graph_manager = AsyncKuzuManager(
#             config.get("graphdb_path", "./rag_main_db")
#         )

#         # ── WeaviateStore جایگزین VectorIndexManager ──
#         self.vector_store = WeaviateStore(
#             collection_name=config.get("weaviate_collection", "Chunks"),
#             mode=ConnectionMode(config.get("weaviate_mode", "docker")),
#             host=config.get("weaviate_host", "localhost"),
#             port=config.get("weaviate_port", 8080),
#             vector_dim=config.get("vector_dim", 1536),
#         )
#         self._vector_ready = False

#     # ── Lifecycle ─────────────────────────────

#     async def startup(self) -> None:
#         """باید یک‌بار در شروع برنامه صدا زده شود."""
#         await self.vector_store.connect()
#         self._vector_ready = True
#         logger.info("WeaviateStore connected")

#     async def shutdown(self) -> None:
#         await self.vector_store.close()
#         self._vector_ready = False

#     # ── Embedding helper ──────────────────────

#     def _embed(self, text: str) -> list[float]:
#         """
#         اینجا embedding model خودت را وصل کن.
#         مثال: OpenAI, sentence-transformers, و غیره
#         """
#         embed_fn = self.config.get("embed_fn")
#         if embed_fn:
#             return embed_fn(text)
#         # fallback موقت - در production حتماً جایگزین کن
#         return [0.0] * self.config.get("vector_dim", 1536)

#     # ── Ingestion ─────────────────────────────

#     async def ingest_text(
#         self,
#         text: str,
#         source: Optional[str] = None,
#         doc_id: Optional[str] = None,
#     ) -> Dict[str, Any]:

#         source = source or "manual"
#         doc_id = doc_id or f"doc_{hash(text) & 0xFFFFFF}"

#         # graph
#         await self.graph_manager.insert_document(
#             doc_id=doc_id,
#             title=source,
#             content=text,
#             source_url=source,
#         )

#         # vector
#         if self._vector_ready:
#             await self.vector_store.upsert(
#                 chunk_id=doc_id,
#                 content=text,
#                 embedding=self._embed(text),
#                 source=source,
#                 doc_id=doc_id,
#             )

#         return {"status": "ok", "source": source, "doc_id": doc_id, "length": len(text)}

#     async def ingest_items(self, items: List[CollectedItem]) -> Dict[str, Any]:
#         inserted = 0

#         for item in items:
#             try:
#                 await self.graph_manager.insert_document(
#                     doc_id=item.id,
#                     title=item.source,
#                     content=item.text,
#                     source_url=item.source,
#                 )

#                 if self._vector_ready:
#                     await self.vector_store.upsert(
#                         chunk_id=item.id,
#                         content=item.text,
#                         embedding=self._embed(item.text),
#                         source=item.source,
#                         doc_id=item.id,
#                         extra_metadata=str(item.metadata),
#                     )

#                 inserted += 1

#             except Exception as e:
#                 logger.warning("Ingest failed %s: %s", item.id, e)

#         return {"inserted": inserted, "total": len(items)}

#     # ── Collect & Ingest ──────────────────────

#     async def collect_and_ingest_news(self, query: str) -> Dict[str, Any]:
#         items = self.data_manager.collect_political_news(query=query)
#         return await self.ingest_items(items)

#     async def collect_and_ingest_social(self, query: str) -> Dict[str, Any]:
#         items = self.data_manager.collect_social(query=query)
#         return await self.ingest_items(items)

#     async def collect_and_ingest_financial(self, tickers: List[str]) -> Dict[str, Any]:
#         items = self.data_manager.collect_financial(tickers)
#         return await self.ingest_items(items)

#     # ── Search (Graph RAG) ────────────────────

#     async def search(
#         self,
#         query: str,
#         top_k: int = 10,
#         alpha: float = 0.5,
#         mmr_lambda: float = 0.6,
#         boost_factor: float = 0.3,
#     ) -> list[dict]:
#         """
#         Graph RAG search:
#         1. entity استخراج از KuzuDB
#         2. entity_weights ساخته می‌شود
#         3. WeaviateStore.graph_rag_search اجرا می‌شود
#         """
#         if not self._vector_ready:
#             logger.warning("Vector store not ready")
#             return []

#         # entity‌های مرتبط از graph
#         entity_result = await self.graph_manager.find_entity_by_name(query, limit=10)
#         entity_weights: dict[str, float] = {}
#         if entity_result and entity_result.rows:
#             for i, row in enumerate(entity_result.rows):
#                 eid = row.get("e.id") or row.get("id", "")
#                 if eid:
#                     # نزدیک‌ترین entity وزن بیشتر دارد
#                     entity_weights[eid] = 1.0 / (i + 1)

#         results = await self.vector_store.graph_rag_search(
#             query_text=query,
#             query_embedding=self._embed(query),
#             entity_weights=entity_weights,
#             top_k=top_k,
#             alpha=alpha,
#             mmr_lambda=mmr_lambda,
#             boost_factor=boost_factor,
#         )

#         return [
#             {
#                 "chunk_id": r.chunk_id,
#                 "content": r.content,
#                 "score": r.score,
#                 "source": r.source,
#                 "entity_ids": r.entity_ids,
#             }
#             for r in results
#         ]

#     # ── Context Manager ───────────────────────

#     async def __aenter__(self):
#         await self.startup()
#         return self

#     async def __aexit__(self, *_):
#         await self.shutdown()
