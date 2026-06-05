# weaviate_store.py
"""
Production-grade Weaviate vector store
Features:
- Connection Factory (Docker/Embedded/Cloud)
- Schema Builder with BM25/HNSW tuning
- Upsert/Batch with error handling
- Vector/BM25/Hybrid/Multi-vector search
- MMR reranking
- Entity boosting
- Multi-tenancy
- Async support
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery, Move

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Enums & Dataclasses
# ─────────────────────────────────────────────

class ConnectionMode(str, Enum):
    DOCKER = "docker"
    EMBEDDED = "embedded"
    CLOUD = "cloud"


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    source: str = ""
    doc_id: str = ""
    entity_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    explain_score: str = ""


@dataclass
class UpsertResult:
    chunk_id: str
    created: bool          # True=insert, False=update
    error: Optional[str] = None


# ─────────────────────────────────────────────
# Connection Factory
# ─────────────────────────────────────────────

class WeaviateConnectionFactory:
    """
    سه حالت اتصال:
    - DOCKER   : weaviate روی localhost یا host دلخواه
    - EMBEDDED : weaviate داخل همین پروسه (بدون Docker)
    - CLOUD    : Weaviate Cloud Services (WCS)
    """

    @staticmethod
    def create(
        mode: ConnectionMode = ConnectionMode.DOCKER,
        host: str = "localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        url: str = "",
        api_key: str = "",
        timeout: tuple[int, int] = (10, 60),
    ) -> weaviate.WeaviateClient:

        if mode == ConnectionMode.DOCKER:
            return weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port,
                additional_config=wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=timeout[0], query=timeout[1])
                ),
            )

        if mode == ConnectionMode.EMBEDDED:
            return weaviate.connect_to_embedded(
                additional_config=wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=timeout[0], query=timeout[1])
                ),
            )

        if mode == ConnectionMode.CLOUD:
            if not url or not api_key:
                raise ValueError("Cloud mode requires url and api_key")
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=wvc.init.Auth.api_key(api_key),
                additional_config=wvc.init.AdditionalConfig(
                    timeout=wvc.init.Timeout(init=timeout[0], query=timeout[1])
                ),
            )

        raise ValueError(f"Unknown mode: {mode}")


# ─────────────────────────────────────────────
# Schema Builder
# ─────────────────────────────────────────────

class CollectionSchemaBuilder:
    """
    Schema با:
    - BM25 tuning (b, k1)
    - HNSW tuning (ef, ef_construction, dynamic_ef)
    - Tokenization per property
    - Named vectors (اختیاری)
    """

    @staticmethod
    def build(
        collection_name: str,
        vector_dim: int = 1536,
        bm25_b: float = 0.75,
        bm25_k1: float = 1.2,
        hnsw_ef: int = 128,
        hnsw_ef_construction: int = 256,
        distance_metric: VectorDistances = VectorDistances.COSINE,
        multi_tenancy: bool = False,
    ) -> dict:
        """
        خروجی: dict آماده برای client.collections.create(**schema)
        """
        properties = [
            Property(
                name="chunk_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,   # exact match
                skip_vectorization=True,
            ),
            Property(
                name="content",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,    # full-text BM25
            ),
            Property(
                name="source",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
            Property(
                name="doc_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
            Property(
                name="entity_ids",
                data_type=DataType.TEXT_ARRAY,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
            Property(
                name="chunk_index",
                data_type=DataType.INT,
                skip_vectorization=True,
            ),
            Property(
                name="token_count",
                data_type=DataType.INT,
                skip_vectorization=True,
            ),
            Property(
                name="language",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
            Property(
                name="created_at",
                data_type=DataType.DATE,
                skip_vectorization=True,
            ),
            Property(
                name="extra_metadata",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                skip_vectorization=True,
            ),
        ]

        vectorizer = Configure.Vectorizer.none()

        vector_index = Configure.VectorIndex.hnsw(
            ef=hnsw_ef,
            ef_construction=hnsw_ef_construction,
            dynamic_ef_min=100,
            dynamic_ef_max=500,
            dynamic_ef_factor=8,
            distance_metric=distance_metric,
            vector_cache_max_objects=500_000,
        )

        inverted_index = Configure.inverted_index(
            bm25_b=bm25_b,
            bm25_k1=bm25_k1,
            index_null_state=True,
            index_property_length=True,
        )

        schema: dict[str, Any] = dict(
            name=collection_name,
            properties=properties,
            vectorizer_config=vectorizer,
            vector_index_config=vector_index,
            inverted_index_config=inverted_index,
        )

        if multi_tenancy:
            schema["multi_tenancy_config"] = Configure.multi_tenancy(enabled=True)

        return schema


# ─────────────────────────────────────────────
# Main Store
# ─────────────────────────────────────────────

class WeaviateStore:
    """
    Production-grade vector store.

    مثال استفاده:
        store = WeaviateStore(collection_name="Chunks")
        await store.connect()
        await store.upsert(chunk_id="c1", content="...", embedding=[...])
        results = await store.hybrid_search("query", embedding=[...])
        await store.close()
    """

    def __init__(
        self,
        collection_name: str = "Chunks",
        mode: ConnectionMode = ConnectionMode.DOCKER,
        host: str = "localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        cloud_url: str = "",
        api_key: str = "",
        vector_dim: int = 1536,
        bm25_b: float = 0.75,
        bm25_k1: float = 1.2,
        hnsw_ef: int = 128,
        hnsw_ef_construction: int = 256,
        batch_size: int = 100,
        multi_tenancy: bool = False,
        timeout: tuple[int, int] = (10, 60),
    ):
        self.collection_name = collection_name
        self.mode = mode
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.cloud_url = cloud_url
        self.api_key = api_key
        self.vector_dim = vector_dim
        self.bm25_b = bm25_b
        self.bm25_k1 = bm25_k1
        self.hnsw_ef = hnsw_ef
        self.hnsw_ef_construction = hnsw_ef_construction
        self.batch_size = batch_size
        self.multi_tenancy = multi_tenancy
        self.timeout = timeout

        self._client: Optional[weaviate.WeaviateClient] = None

    # ── Lifecycle ──────────────────────────────

    async def connect(self) -> None:
        self._client = WeaviateConnectionFactory.create(
            mode=self.mode,
            host=self.host,
            port=self.port,
            grpc_port=self.grpc_port,
            url=self.cloud_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
        await self._ensure_collection()
        logger.info("WeaviateStore connected: %s", self.collection_name)

    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    async def health_check(self) -> bool:
        try:
            return self._client is not None and self._client.is_ready()
        except Exception:
            return False

    # ── Schema ────────────────────────────────

    async def _ensure_collection(self) -> None:
        if self._client.collections.exists(self.collection_name):
            logger.debug("Collection already exists: %s", self.collection_name)
            return

        schema = CollectionSchemaBuilder.build(
            collection_name=self.collection_name,
            vector_dim=self.vector_dim,
            bm25_b=self.bm25_b,
            bm25_k1=self.bm25_k1,
            hnsw_ef=self.hnsw_ef,
            hnsw_ef_construction=self.hnsw_ef_construction,
            multi_tenancy=self.multi_tenancy,
        )
        self._client.collections.create(**schema)
        logger.info("Collection created: %s", self.collection_name)

    def _collection(self, tenant: Optional[str] = None):
        col = self._client.collections.get(self.collection_name)
        if tenant:
            return col.with_tenant(tenant)
        return col

    # ── Ingestion ─────────────────────────────

    async def upsert(
        self,
        chunk_id: str,
        content: str,
        embedding: list[float],
        source: str = "",
        doc_id: str = "",
        entity_ids: Optional[list[str]] = None,
        chunk_index: int = 0,
        token_count: int = 0,
        language: str = "fa",
        extra_metadata: str = "",
        tenant: Optional[str] = None,
    ) -> UpsertResult:
        obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        props = {
            "chunk_id": chunk_id,
            "content": content,
            "source": source,
            "doc_id": doc_id,
            "entity_ids": entity_ids or [],
            "chunk_index": chunk_index,
            "token_count": token_count,
            "language": language,
            "extra_metadata": extra_metadata,
        }
        try:
            col = self._collection(tenant)
            exists = col.data.exists(obj_uuid)
            if exists:
                col.data.update(uuid=obj_uuid, properties=props, vector=embedding)
                return UpsertResult(chunk_id=chunk_id, created=False)
            else:
                col.data.insert(uuid=obj_uuid, properties=props, vector=embedding)
                return UpsertResult(chunk_id=chunk_id, created=True)
        except Exception as e:
            logger.error("Upsert failed for %s: %s", chunk_id, e)
            return UpsertResult(chunk_id=chunk_id, created=False, error=str(e))

    async def upsert_batch(
        self,
        items: list[dict],
        tenant: Optional[str] = None,
    ) -> list[UpsertResult]:
        """
        items: list of dicts با همان پارامترهای upsert
        """
        results: list[UpsertResult] = []
        for i in range(0, len(items), self.batch_size):
            chunk = items[i : i + self.batch_size]
            for item in chunk:
                r = await self.upsert(**item, tenant=tenant)
                results.append(r)
            logger.debug("Batch progress: %d/%d", min(i + self.batch_size, len(items)), len(items))
        return results

    # ── Search ────────────────────────────────

    def _parse_results(self, objects) -> list[SearchResult]:
        out = []
        for o in objects:
            p = o.properties
            score = 0.0
            explain = ""
            if o.metadata:
                score = (
                    o.metadata.score
                    or o.metadata.certainty
                    or o.metadata.distance
                    or 0.0
                )
                explain = getattr(o.metadata, "explain_score", "") or ""
            out.append(
                SearchResult(
                    chunk_id=p.get("chunk_id", ""),
                    content=p.get("content", ""),
                    score=float(score),
                    source=p.get("source", ""),
                    doc_id=p.get("doc_id", ""),
                    entity_ids=p.get("entity_ids", []),
                    metadata={k: v for k, v in p.items() if k not in ("chunk_id", "content", "source", "doc_id", "entity_ids")},
                    embedding=list(o.vector.get("default", [])) if o.vector else [],
                    explain_score=explain,
                )
            )
        return out

    async def vector_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        certainty: Optional[float] = None,
        distance: Optional[float] = None,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
    ) -> list[SearchResult]:
        col = self._collection(tenant)
        kwargs: dict[str, Any] = dict(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(score=True, certainty=True, distance=True),
            return_properties=True,
            include_vector=True,
        )
        if certainty is not None:
            kwargs["near_vector"] = wvc.query.NearVectorInputType(
                vector=query_embedding, certainty=certainty
            )
        if distance is not None:
            kwargs["near_vector"] = wvc.query.NearVectorInputType(
                vector=query_embedding, distance=distance
            )
        if filters:
            kwargs["filters"] = filters
        res = col.query.near_vector(**kwargs)
        return self._parse_results(res.objects)

    async def bm25_search(
        self,
        query_text: str,
        top_k: int = 10,
        properties: Optional[list[str]] = None,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
    ) -> list[SearchResult]:
        col = self._collection(tenant)
        kwargs: dict[str, Any] = dict(
            query=query_text,
            limit=top_k,
            query_properties=properties or ["content"],
            return_metadata=MetadataQuery(score=True),
            return_properties=True,
            include_vector=True,
        )
        if filters:
            kwargs["filters"] = filters
        res = col.query.bm25(**kwargs)
        return self._parse_results(res.objects)

    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
        fusion_type: HybridFusion = HybridFusion.RELATIVE_SCORE,
        filters: Optional[Filter] = None,
        tenant: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        alpha=1.0 → pure vector
        alpha=0.0 → pure BM25
        alpha=0.5 → balanced
        """
        print(f"{query_text}")
        print(f"*"*50)
        print(f"{query_embedding['embedding']}")
        col = self._collection(tenant)
        kwargs: dict[str, Any] = dict(
            query=query_text,
            vector=query_embedding['embedding'],
            limit=top_k,
            alpha=alpha,
            fusion_type=fusion_type,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            return_properties=True,
            include_vector=True,
        )
        if filters:
            kwargs["filters"] = filters
        res = col.query.hybrid(**kwargs)
        return self._parse_results(res.objects)

    async def search_by_entity_ids(
        self,
        entity_ids: list[str],
        query_embedding: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
        query_text: str = "",
        tenant: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        جستجو محدود به chunk‌هایی که entity_ids مشخص دارند.
        برای Graph RAG: ابتدا KuzuDB entity پیدا می‌کند، سپس اینجا جستجو می‌شود.
        """
        entity_filter = Filter.by_property("entity_ids").contains_any(entity_ids)
        if query_text:
            return await self.hybrid_search(
                query_text=query_text,
                query_embedding=query_embedding,
                top_k=top_k,
                alpha=alpha,
                filters=entity_filter,
                tenant=tenant,
            )
        return await self.vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=entity_filter,
            tenant=tenant,
        )

    # ── MMR Reranking ─────────────────────────

    async def mmr_rerank(
        self,
        results: list[SearchResult],
        query_embedding: list[float],
        lambda_param: float = 0.5,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Maximal Marginal Relevance reranking.

        فرمول:
            MMR(d) = λ · sim(d, query) − (1−λ) · max_{s∈S} sim(d, s)

        lambda_param=1.0 → فقط relevance (مثل جستجوی معمولی)
        lambda_param=0.0 → فقط diversity (حداکثر تنوع)
        lambda_param=0.5 → تعادل بین relevance و diversity

        نیاز: SearchResult.embedding پر باشد (include_vector=True در جستجو)
        """
        if len(results) <= 1:
            return results[:top_k]

        # فیلتر نتایج بدون embedding
        valid = [r for r in results if r.embedding]
        if not valid:
            logger.warning("MMR: no embeddings found, returning original order")
            return results[:top_k]

        query_vec = np.array(query_embedding, dtype=np.float32)

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        # شباهت هر نتیجه به query
        doc_vecs = [np.array(r.embedding, dtype=np.float32) for r in valid]
        query_sims = [cosine(query_vec, v) for v in doc_vecs]

        selected: list[SearchResult] = []
        selected_vecs: list[np.ndarray] = []
        remaining_idx = list(range(len(valid)))

        # اولین: بیشترین شباهت به query
        first = int(np.argmax(query_sims))
        selected.append(valid[first])
        selected_vecs.append(doc_vecs[first])
        remaining_idx.remove(first)

        while len(selected) < min(top_k, len(valid)):
            best_score = -np.inf
            best_i = remaining_idx[0]

            for i in remaining_idx:
                rel = query_sims[i]
                max_sim = max(cosine(doc_vecs[i], sv) for sv in selected_vecs)
                mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_i = i

            selected.append(valid[best_i])
            selected_vecs.append(doc_vecs[best_i])
            remaining_idx.remove(best_i)

        return selected

    # ── Entity Boosting ───────────────────────

    async def entity_boost(
        self,
        results: list[SearchResult],
        entity_weights: dict[str, float],
        boost_factor: float = 0.3,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Entity boosting: score نتایج را بر اساس وزن entity‌های مرتبط افزایش می‌دهد.

        فرمول:
            boosted_score = score × (1 + boost_factor × Σ entity_weight)

        entity_weights: {entity_id: weight}
            مثال: {"person_123": 1.0, "org_456": 0.7}
            این وزن‌ها معمولاً از Graph traversal در KuzuDB می‌آیند.

        boost_factor: ضریب کلی تأثیر boost (0.0 تا 1.0)
            0.0 → بدون تأثیر
            1.0 → تأثیر کامل
        """
        boosted: list[SearchResult] = []

        for result in results:
            total_boost = sum(
                entity_weights.get(eid, 0.0)
                for eid in result.entity_ids
            )
            new_score = result.score * (1 + boost_factor * total_boost)
            new_score = min(new_score, 1.0)  # cap at 1.0

            boosted.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=new_score,
                    source=result.source,
                    doc_id=result.doc_id,
                    entity_ids=result.entity_ids,
                    metadata=result.metadata,
                    embedding=result.embedding,
                    explain_score=result.explain_score,
                )
            )

        return sorted(boosted, key=lambda x: x.score, reverse=True)[:top_k]

    # ── Graph RAG Pipeline ────────────────────

    async def graph_rag_search(
        self,
        query_text: str,
        query_embedding: list[float],
        entity_weights: dict[str, float],
        top_k: int = 10,
        alpha: float = 0.5,
        mmr_lambda: float = 0.6,
        boost_factor: float = 0.3,
        candidate_multiplier: int = 3,
        tenant: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Pipeline کامل Graph RAG:

        1. Hybrid search با candidate_multiplier × top_k نتیجه
        2. Entity boosting بر اساس وزن‌های KuzuDB
        3. MMR reranking برای diversity
        4. بازگشت top_k نتیجه نهایی

        entity_weights از KuzuDB می‌آید:
            kuzu_results = kuzu_client.find_related_entities(query_entities)
            entity_weights = {e.id: e.relevance_score for e in kuzu_results}
        """
        candidates = await self.hybrid_search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k * candidate_multiplier,
            alpha=alpha,
            tenant=tenant,
        )

        if entity_weights:
            candidates = await self.entity_boost(
                results=candidates,
                entity_weights=entity_weights,
                boost_factor=boost_factor,
                top_k=top_k * candidate_multiplier,
            )

        final = await self.mmr_rerank(
            results=candidates,
            query_embedding=query_embedding,
            lambda_param=mmr_lambda,
            top_k=top_k,
        )

        return final

    # ── Delete ────────────────────────────────

    async def delete(self, chunk_id: str, tenant: Optional[str] = None) -> bool:
        try:
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            self._collection(tenant).data.delete_by_id(obj_uuid)
            return True
        except Exception as e:
            logger.error("Delete failed for %s: %s", chunk_id, e)
            return False

    async def delete_by_doc_id(self, doc_id: str, tenant: Optional[str] = None) -> int:
        col = self._collection(tenant)
        res = col.data.delete_many(
            where=Filter.by_property("doc_id").equal(doc_id)
        )
        return res.successful

    # ── Context Manager ───────────────────────

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()
