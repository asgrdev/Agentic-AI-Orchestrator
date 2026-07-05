# retriever_agent.py

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
from vector_store.weaviate_client import ConnectionMode, WeaviateStore
from agents.state import AgentState, RetrievalContext
from ingestion.embedding_generator import Qwen3EmbeddingClient
from core.model_wrapper import get_embedding_model
from core.memory_monitor import get_memory_monitor

logger = logging.getLogger(__name__)

# حداکثر entity برای subgraph
_MAX_ENTITIES = 15
# حداکثر موازی‌سازی روی KuzuDB
_KUZU_CONCURRENCY = 4


class RetrieverAgent:

    def __init__(self, config: dict) -> None:
        # استفاده از Model Wrapper برای مدیریت بهتر حافظه
        self._use_model_manager = config.get("use_model_manager", True)
        
        if self._use_model_manager:
            # استفاده از Model Manager
            self._embed_wrapper = get_embedding_model()
            self._embed_fn = None  # lazy load
        else:
            # روش قدیمی
            if "embed_client_factory" in config:
                embed_client = config["embed_client_factory"]()
            else:
                embed_client = config["embed_client"]
            self._embed_fn: Qwen3EmbeddingClient = embed_client
            self._embed_wrapper = None
        
        self._memory_monitor = get_memory_monitor()

        self._kuzu = AsyncKuzuManager(config["kuzu_path"])

        wv = config.get("weaviate", {})
        self._weaviate = WeaviateStore(
            collection_name=config.get("collection_name", "Chunks"),
            mode=ConnectionMode(wv.get("mode", "docker")),
            host=wv.get("host", "localhost"),
            port=wv.get("port", 8080),
            grpc_port=wv.get("grpc_port", 50051),
            cloud_url=wv.get("cloud_url", ""),
            api_key=wv.get("api_key", ""),
            vector_dim=wv.get("vector_dim", 1536),
        )

        self._top_k      = config.get("vector_top_k", 10)
        self._alpha      = config.get("hybrid_alpha", 0.7)
        self._mmr_lambda = config.get("mmr_lambda", 0.6)
        self._boost      = config.get("boost_factor", 0.3)
        self._hops       = config.get("graph_hops", 2)
        self._conf_top_k = config.get("confidence_top_k", 3)

        # semaphore برای محدود کردن موازی‌سازی KuzuDB
        self._kuzu_sem = asyncio.Semaphore(_KUZU_CONCURRENCY)
        self._ready = False

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def startup(self) -> None:
        if self._ready:
            return
        await self._kuzu.__aenter__()
        await self._weaviate.connect()
        self._ready = True
        logger.info("RetrieverAgent ready.")

    async def shutdown(self) -> None:
        await self._kuzu.__aexit__(None, None, None)
        await self._weaviate.close()
        self._ready = False

    # ── Main Entry Point ───────────────────────────────────────────────

    async def retrieve(self, state: AgentState) -> None:
        if not self._ready:
            raise RuntimeError("Call startup() first")

        # همه مراحل موازی که می‌توانند
        embedding, entity_weights = await asyncio.gather(
            self._embed(state.query),
            self._build_entity_weights(state),
        )
        # iteration را orchestrator مدیریت می‌کند (شمارش چرخه، نه شمارش retrieve)

        # vector search با entity weights آماده
        search_results = await self._weaviate.graph_rag_search(
            query_text=state.query,
            query_embedding=embedding,
            entity_weights=entity_weights,
            top_k=self._top_k,
            alpha=self._alpha,
            mmr_lambda=self._mmr_lambda,
            boost_factor=self._boost,
        )

        logger.debug("Weaviate returned %d results", len(search_results))

        # entity_ids از نتایج weaviate + entity_weights ترکیب می‌شوند
        entity_ids = self._merge_entity_ids(search_results, entity_weights)

        # subgraph با semaphore
        graph_nodes, graph_edges = await self._fetch_subgraph(entity_ids)

        state.retrieval = RetrievalContext(
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
            vector_chunks=[self._result_to_dict(r) for r in search_results],
            bm25_docs=[],
            fused_context="",   # ReasonerAgent این را می‌سازد
            confidence=self._compute_confidence(search_results),
        )

        logger.info(
            "Retrieval done | chunks=%d nodes=%d edges=%d confidence=%.2f",
            len(search_results),
            len(graph_nodes),
            len(graph_edges),
            state.retrieval.confidence,
        )

    # ── Private Helpers ────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding with memory management"""
        if self._use_model_manager and self._embed_wrapper:
            # استفاده از Model Manager
            def _embed_sync():
                with self._embed_wrapper.use_model() as model:
                    result = model.embed_single(text)
                    return result["embedding"]
            
            return await asyncio.get_running_loop().run_in_executor(
                None, _embed_sync
            )
        elif self._embed_fn:
            # روش قدیمی
            result = await asyncio.get_running_loop().run_in_executor(
                None, self._embed_fn.embed_single, text
            )
            return result["embedding"]
        else:
            logger.warning("No embed_fn — zero vector used")
            return [0.0] * 1536

    async def _build_entity_weights(self, state: AgentState) -> dict[str, float]:
        """
        entity_weights نرمالیزه‌شده از دو منبع:
        - extracted_entities از phi4  → وزن 1.0 تا 0.5
        - KuzuDB name search          → وزن 0.4 تا 0.1
        نرمالیزه به [0, 1] در انتها
        """
        weights: dict[str, float] = {}

        # منبع 1: phi4 extracted entities — planner کلید "text" می‌دهد
        for i, entity in enumerate(state.extracted_entities):
            eid = entity.get("id") or entity.get("name") or entity.get("text", "")
            if eid:
                # decay از 1.0 به 0.5
                weights[eid] = max(1.0 - i * 0.1, 0.5)

        # منبع 2: KuzuDB - فقط اگر extracted_entities کم باشد
        if len(weights) < 3:
            try:
                result = await self._kuzu.find_entity_by_name(state.query, limit=5)
                if result and result.rows:
                    for i, row in enumerate(result.rows):
                        eid = row.get("e.id") or row.get("id", "")
                        if eid and eid not in weights:
                            # وزن پایین‌تر و decay سریع‌تر
                            weights[eid] = max(0.4 - i * 0.06, 0.1)
            except Exception as e:
                logger.warning("KuzuDB entity lookup failed: %s", e)

        # normalize به [0, 1]
        if weights:
            max_w = max(weights.values())
            if max_w > 0:
                weights = {k: v / max_w for k, v in weights.items()}

        logger.debug("Entity weights: %d entities", len(weights))
        return weights

    async def _fetch_subgraph(
        self, entity_ids: list[str]
    ) -> tuple[list[dict], list[dict]]:
        """
        با semaphore از اشباع thread pool جلوگیری می‌کند.
        """
        nodes: list[dict] = []
        edges: list[dict] = []
        seen_nodes: set[str] = set()
        seen_edges: set[str] = set()

        async def fetch_one(eid: str) -> Any:
            async with self._kuzu_sem:
                return await self._kuzu.get_neighbors(eid, hops=self._hops)

        tasks = [fetch_one(eid) for eid in entity_ids[:_MAX_ENTITIES]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for eid, result in zip(entity_ids[:_MAX_ENTITIES], results):
            if isinstance(result, Exception):
                logger.warning("get_neighbors failed for %s: %s", eid, result)
                continue
            if not result or not result.rows:
                continue

            for row in result.rows:
                node_id = row.get("n.id") or row.get("id", "")
                if node_id and node_id not in seen_nodes:
                    nodes.append({
                        "id":         node_id,
                        "name":       row.get("n.name") or row.get("name", ""),
                        "type":       row.get("n.type") or row.get("type", ""),
                        "properties": {k: v for k, v in row.items() if not k.startswith("r.")},
                    })
                    seen_nodes.add(node_id)

                edge_key = f"{eid}->{node_id}"
                if edge_key not in seen_edges:
                    edges.append({
                        "source":     eid,
                        "target":     node_id,
                        "type":       row.get("r.type") or row.get("rel_type", "RELATED"),
                        "properties": {k: v for k, v in row.items() if k.startswith("r.")},
                    })
                    seen_edges.add(edge_key)

        return nodes, edges

    def _merge_entity_ids(
        self,
        search_results: list[Any],
        entity_weights: dict[str, float],
    ) -> list[str]:
        """
        entity_ids از Weaviate results + entity_weights از KuzuDB
        مرتب‌شده بر اساس وزن
        """
        ids: dict[str, float] = dict(entity_weights)  # از KuzuDB

        for result in search_results:
            for eid in getattr(result, "entity_ids", []):
                if eid and eid not in ids:
                    # entity جدید از weaviate با وزن پایه score نتیجه
                    ids[eid] = getattr(result, "score", 0.1)

        # مرتب بر اساس وزن، مهم‌ترین‌ها اول
        return [k for k, _ in sorted(ids.items(), key=lambda x: x[1], reverse=True)]

    def _compute_confidence(self, search_results: list[Any]) -> float:
        if not search_results:
            return 0.0
        scores = sorted(
            [getattr(r, "score", 0.0) for r in search_results],
            reverse=True,
        )[: self._conf_top_k]
        return round(sum(scores) / len(scores), 4)

    @staticmethod
    def _result_to_dict(result: Any) -> dict:
        return {
            "chunk_id":   getattr(result, "chunk_id", ""),
            "content":    getattr(result, "content", ""),
            "score":      getattr(result, "score", 0.0),
            "source":     getattr(result, "source", ""),
            "doc_id":     getattr(result, "doc_id", ""),
            "entity_ids": getattr(result, "entity_ids", []),
            "metadata":   getattr(result, "metadata", {}),
        }



# # retriever_agent.py
# """
# RetrieverAgent - مرحله RETRIEVE در orchestrator

# Pipeline:
#     AgentState (query + extracted_entities)
#         ↓
#     [1] find_entity_by_name() در KuzuDB → entity_weights
#         ↓
#     [2] WeaviateStore.graph_rag_search() → SearchResults (hybrid + boost + MMR)
#         ↓
#     [3] AsyncKuzuManager.get_neighbors() → graph_nodes + graph_edges
#         ↓
#     RetrievalContext پر شده در AgentState
# """

# from __future__ import annotations

# import asyncio
# import logging
# from typing import Any, Callable, Optional

# from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
# from vector_store.weaviate_client import ConnectionMode, WeaviateStore
# from state import AgentState, RetrievalContext

# logger = logging.getLogger(__name__)


# class RetrieverAgent:
#     """
#     مسئول بازیابی اطلاعات از Weaviate + KuzuDB.
#     هیچ LLM reasoning انجام نمی‌دهد.
#     """

#     def __init__(self, config: dict) -> None:
#         """
#         config keys:
#             kuzu_path       : str  - مسیر KuzuDB
#             weaviate        : dict - تنظیمات WeaviateStore
#             embed_fn        : Callable[[str], list[float]] - تابع embedding
#             collection_name : str  - نام collection در Weaviate (پیش‌فرض "Chunks")
#             vector_top_k    : int  - تعداد نتایج vector search (پیش‌فرض 10)
#             hybrid_alpha    : float - 0=BM25, 1=vector (پیش‌فرض 0.7)
#             mmr_lambda      : float - 0=diversity, 1=relevance (پیش‌فرض 0.6)
#             boost_factor    : float - ضریب entity boost (پیش‌فرض 0.3)
#             graph_hops      : int  - عمق k-hop در graph (پیش‌فرض 2)
#             confidence_top_k: int  - تعداد نتایج برای محاسبه confidence (پیش‌فرض 3)
#         """
#         self._embed_fn: Optional[Callable[[str], list[float]]] = config.get("embed_fn")

#         # KuzuDB
#         self._kuzu = AsyncKuzuManager(config["kuzu_path"])

#         # Weaviate - ساخت از dict config
#         wv_cfg = config.get("weaviate", {})
#         self._weaviate = WeaviateStore(
#             collection_name=config.get("collection_name", "Chunks"),
#             mode=ConnectionMode(wv_cfg.get("mode", "docker")),
#             host=wv_cfg.get("host", "localhost"),
#             port=wv_cfg.get("port", 8080),
#             grpc_port=wv_cfg.get("grpc_port", 50051),
#             cloud_url=wv_cfg.get("cloud_url", ""),
#             api_key=wv_cfg.get("api_key", ""),
#             vector_dim=wv_cfg.get("vector_dim", 1536),
#         )

#         # پارامترهای جستجو
#         self._top_k       = config.get("vector_top_k", 10)
#         self._alpha       = config.get("hybrid_alpha", 0.7)
#         self._mmr_lambda  = config.get("mmr_lambda", 0.6)
#         self._boost       = config.get("boost_factor", 0.3)
#         self._hops        = config.get("graph_hops", 2)
#         self._conf_top_k  = config.get("confidence_top_k", 3)

#         self._ready = False

#     # ── Lifecycle ──────────────────────────────────────────────────────

#     async def startup(self) -> None:
#         await self._kuzu.__aenter__()
#         await self._weaviate.connect()
#         self._ready = True
#         logger.info("RetrieverAgent ready.")

#     async def shutdown(self) -> None:
#         await self._kuzu.__aexit__(None, None, None)
#         await self._weaviate.close()
#         self._ready = False

#     # ── Main Entry Point ───────────────────────────────────────────────

#     async def retrieve(self, state: AgentState) -> None:
#         """
#         نتایج را مستقیماً در state.retrieval می‌نویسد.
#         هیچ مقداری return نمی‌کند.
#         """
#         if not self._ready:
#             raise RuntimeError("Call startup() first")

#         query = state.query

#         # ── Step 1: embedding ──────────────────────────────────────────
#         embedding = await self._embed(query)

#         # ── Step 2: entity weights از KuzuDB ──────────────────────────
#         entity_weights = await self._build_entity_weights(state)

#         # ── Step 3: Graph RAG search در Weaviate ──────────────────────
#         search_results = await self._weaviate.graph_rag_search(
#             query_text=query,
#             query_embedding=embedding,
#             entity_weights=entity_weights,
#             top_k=self._top_k,
#             alpha=self._alpha,
#             mmr_lambda=self._mmr_lambda,
#             boost_factor=self._boost,
#         )

#         logger.debug("Weaviate returned %d results", len(search_results))

#         # ── Step 4: graph neighbors از KuzuDB ─────────────────────────
#         entity_ids = self._extract_entity_ids(search_results)
#         graph_nodes, graph_edges = await self._fetch_subgraph(entity_ids)

#         # ── Step 5: fused context برای ReasonerAgent ──────────────────
#         fused = self._build_fused_context(search_results)

#         # ── Step 6: confidence score ───────────────────────────────────
#         confidence = self._compute_confidence(search_results)

#         # ── نوشتن در state ─────────────────────────────────────────────
#         state.retrieval = RetrievalContext(
#             graph_nodes=graph_nodes,
#             graph_edges=graph_edges,
#             vector_chunks=[self._result_to_dict(r) for r in search_results],
#             bm25_docs=[],          # در صورت نیاز می‌توان BM25 جداگانه اضافه کرد
#             fused_context=fused,
#             confidence=confidence,
#         )

#         logger.info(
#             "Retrieval done | chunks=%d nodes=%d edges=%d confidence=%.2f",
#             len(search_results),
#             len(graph_nodes),
#             len(graph_edges),
#             confidence,
#         )

#     # ── Private Helpers ────────────────────────────────────────────────

#     async def _embed(self, text: str) -> list[float]:
#         """embedding را در thread pool اجرا می‌کند."""
#         if not self._embed_fn:
#             logger.warning("No embed_fn provided, using zero vector")
#             return [0.0] * 1536
#         return await asyncio.get_running_loop().run_in_executor(
#             None, self._embed_fn, text
#         )

#     async def _build_entity_weights(self, state: AgentState) -> dict[str, float]:
#         """
#         entity_weights از دو منبع:
#         1. extracted_entities که phi4 پیدا کرده (وزن بالاتر)
#         2. جستجوی مستقیم نام query در KuzuDB (وزن پایین‌تر)
#         """
#         weights: dict[str, float] = {}

#         # از extracted_entities که phi4 استخراج کرده
#         for i, entity in enumerate(state.extracted_entities):
#             eid = entity.get("id") or entity.get("name", "")
#             if not eid:
#                 continue
#             # اولین entity مهم‌ترین است
#             weights[eid] = 1.0 / (i + 1)

#         # جستجوی اضافه در KuzuDB با نام query
#         try:
#             result = await self._kuzu.find_entity_by_name(state.query, limit=5)
#             if result and result.rows:
#                 for i, row in enumerate(result.rows):
#                     eid = row.get("e.id") or row.get("id", "")
#                     if eid and eid not in weights:
#                         # وزن پایین‌تر از extracted_entities
#                         weights[eid] = 0.5 / (i + 1)
#         except Exception as e:
#             logger.warning("KuzuDB entity lookup failed: %s", e)

#         logger.debug("Entity weights: %d entities", len(weights))
#         return weights

#     async def _fetch_subgraph(
#         self, entity_ids: list[str]
#     ) -> tuple[list[dict], list[dict]]:
#         """
#         برای هر entity، neighbors را می‌گیرد و نتایج را ادغام می‌کند.
#         از get_neighbors() که در AsyncKuzuManager موجود است استفاده می‌کند.
#         """
#         nodes: list[dict] = []
#         edges: list[dict] = []
#         seen_nodes: set[str] = set()
#         seen_edges: set[str] = set()

#         # همه entity‌ها را موازی fetch می‌کنیم
#         tasks = [
#             self._kuzu.get_neighbors(eid, hops=self._hops)
#             for eid in entity_ids[:20]  # حداکثر 20 entity برای جلوگیری از overload
#         ]

#         results = await asyncio.gather(*tasks, return_exceptions=True)

#         for eid, result in zip(entity_ids[:20], results):
#             if isinstance(result, Exception):
#                 logger.warning("get_neighbors failed for %s: %s", eid, result)
#                 continue
#             if not result or not result.rows:
#                 continue

#             for row in result.rows:
#                 # node
#                 node_id = row.get("n.id") or row.get("id", "")
#                 if node_id and node_id not in seen_nodes:
#                     nodes.append({
#                         "id": node_id,
#                         "name": row.get("n.name") or row.get("name", ""),
#                         "type": row.get("n.type") or row.get("type", ""),
#                         "properties": {
#                             k: v for k, v in row.items()
#                             if not k.startswith("r.")
#                         },
#                     })
#                     seen_nodes.add(node_id)

#                 # edge
#                 edge_key = f"{eid}->{node_id}"
#                 if edge_key not in seen_edges:
#                     edges.append({
#                         "source": eid,
#                         "target": node_id,
#                         "type": row.get("r.type") or row.get("rel_type", "RELATED"),
#                         "properties": {
#                             k: v for k, v in row.items()
#                             if k.startswith("r.")
#                         },
#                     })
#                     seen_edges.add(edge_key)

#         return nodes, edges

#     def _extract_entity_ids(self, search_results: list[Any]) -> list[str]:
#         """entity_ids را از نتایج Weaviate استخراج می‌کند."""
#         ids: list[str] = []
#         seen: set[str] = set()
#         for result in search_results:
#             for eid in getattr(result, "entity_ids", []):
#                 if eid and eid not in seen:
#                     ids.append(eid)
#                     seen.add(eid)
#         return ids

#     def _build_fused_context(self, search_results: list[Any]) -> str:
#         """
#         متن یکپارچه از نتایج جستجو برای ReasonerAgent.
#         هر chunk با شماره و منبع مشخص می‌شود.
#         """
#         parts = []
#         for i, r in enumerate(search_results, 1):
#             source = getattr(r, "source", "")
#             content = getattr(r, "content", "")
#             score = getattr(r, "score", 0.0)
#             source_label = f" [{source}]" if source else ""
#             parts.append(f"[{i}]{source_label} (score={score:.2f})\n{content}")
#         return "\n\n---\n\n".join(parts)

#     def _compute_confidence(self, search_results: list[Any]) -> float:
#         """
#         میانگین score بالاترین N نتیجه.
#         اگر نتیجه‌ای نباشد → 0.0
#         """
#         if not search_results:
#             return 0.0
#         scores = sorted(
#             [getattr(r, "score", 0.0) for r in search_results],
#             reverse=True,
#         )[: self._conf_top_k]
#         return round(sum(scores) / len(scores), 4)

#     @staticmethod
#     def _result_to_dict(result: Any) -> dict:
#         """SearchResult dataclass را به dict تبدیل می‌کند."""
#         return {
#             "chunk_id":   getattr(result, "chunk_id", ""),
#             "content":    getattr(result, "content", ""),
#             "score":      getattr(result, "score", 0.0),
#             "source":     getattr(result, "source", ""),
#             "doc_id":     getattr(result, "doc_id", ""),
#             "entity_ids": getattr(result, "entity_ids", []),
#             "metadata":   getattr(result, "metadata", {}),
#         }
