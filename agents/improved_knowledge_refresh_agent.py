"""
Improved Knowledge Refresh Agent - بهبود یافته با گراف معنایی

تغییرات نسبت به نسخه قبل:
1. استفاده از SemanticGraphBuilder برای ساخت گراف معنایی
2. بروزرسانی تدریجی (incremental update) به جای بازنویسی کامل
3. حل تعارض هوشمند
4. نسخه‌بندی دانش
5. متریک‌های کیفیت گراف
6. استفاده از Model Manager برای مدیریت حافظه
"""
from __future__ import annotations

import asyncio
import logging
import re
from time import time
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from agents.state import AgentState, FlowStep
from external_sources.data_collector.manager import DataCollectorManager
from external_sources.data_collector.models import DataDomain
from ingestion.chunk_ingestion_pipeline import ChunkIngestionPipeline
from ingestion.semantic_graph_builder import SemanticGraphBuilder, GraphUpdate
from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
from vector_store.weaviate_client import WeaviateStore
from ingestion.embedding_generator import Qwen3EmbeddingClient
from ingestion.EntityExtractionPipeline import EntityExtractionPipeline
from ingestion.relation_pipeline import RelationPipeline
from core.model_wrapper import get_embedding_model
from core.memory_monitor import get_memory_monitor

logger = logging.getLogger(__name__)


@dataclass
class RefreshMetrics:
    """متریک‌های بروزرسانی دانش"""
    total_items: int = 0
    chunks_ingested: int = 0
    nodes_added: int = 0
    nodes_updated: int = 0
    edges_added: int = 0
    edges_updated: int = 0
    inferred_relations: int = 0
    conflicts_resolved: int = 0
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "total_items": self.total_items,
            "chunks_ingested": self.chunks_ingested,
            "nodes_added": self.nodes_added,
            "nodes_updated": self.nodes_updated,
            "edges_added": self.edges_added,
            "edges_updated": self.edges_updated,
            "inferred_relations": self.inferred_relations,
            "conflicts_resolved": self.conflicts_resolved,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


# نگاشت domain → متد collect در DataCollectorManager
_DOMAIN_COLLECTORS = {
    DataDomain.KNOWLEDGE:   "collect_knowledge",
    DataDomain.SCIENTIFIC:  "collect_scientific",
    DataDomain.NEWS:        "collect_political_news",
    DataDomain.SOCIAL:      "collect_social",
}


class ImprovedKnowledgeRefreshAgent:
    """
    Agent بهبود یافته برای بروزرسانی دانش با قابلیت‌های پیشرفته:
    - گراف معنایی
    - بروزرسانی تدریجی
    - حل تعارض
    - نسخه‌بندی
    """

    def __init__(self, config: dict):
        # استفاده از Model Manager برای مدیریت بهتر حافظه
        self._use_model_manager = config.get("use_model_manager", True)
        
        if self._use_model_manager:
            self._embed_wrapper = get_embedding_model()
            self._embed_client = None
        else:
            if "embed_client_factory" in config:
                embed_client = config["embed_client_factory"]()
            else:
                embed_client = config["embed_client"]
            self._embed_client: Qwen3EmbeddingClient = embed_client
            self._embed_wrapper = None
        
        self._memory_monitor = get_memory_monitor()

        # Data collector
        self.data_manager: DataCollectorManager = config.get(
            "data_manager", DataCollectorManager()
        )

        # KuzuDB & Weaviate
        self._kuzu = AsyncKuzuManager(config["kuzu_path"])
        
        wv = config.get("weaviate", {})
        vector_dim = config.get("vector_dim", 1536)
        if self._use_model_manager and self._embed_wrapper:
            # گرفتن dimension از model wrapper
            with self._embed_wrapper.use_model() as model:
                vector_dim = model.embedding_dim
        elif self._embed_client:
            vector_dim = self._embed_client.embedding_dim
            
        self._weaviate = WeaviateStore(
            collection_name=config.get("collection_name", "Chunks"),
            mode=wv.get("mode", "docker"),
            host=wv.get("host", "localhost"),
            port=wv.get("port", 8080),
            grpc_port=wv.get("grpc_port", 50051),
            cloud_url=wv.get("cloud_url", ""),
            api_key=wv.get("api_key", ""),
            vector_dim=vector_dim,
        )

        # Semantic Graph Builder
        self._semantic_builder = SemanticGraphBuilder(
            graph_manager=self._kuzu,
            enable_inference=config.get("enable_inference", True),
            enable_versioning=config.get("enable_versioning", True),
            confidence_threshold=config.get("confidence_threshold", 0.5),
        )

        # Entity & Relation Pipelines
        entity_config = {
            "ner_model": config.get("ner_model", "dslim/bert-base-NER"),
            "entity_linking_endpoint": config.get("entity_linking_endpoint"),
            "max_workers": config.get("max_workers", 4),
        }
        self._entity_pipeline = EntityExtractionPipeline(entity_config)

        relation_config = {
            "llm_endpoint": config.get("llm_endpoint"),
            "model_name": config.get("relation_model2", "mlx-community/Llama-3.2-3B-Instruct-4bit"),
        }
        self._relation_pipeline = RelationPipeline(relation_config)

        # Chunk Ingestion Pipeline (با semantic builder)
        self._chunk_pipeline = ChunkIngestionPipeline(
            weaviate=self._weaviate,
            kuzu=self._kuzu,
            embed_fn=self._make_embed_fn(),
            entity_pipeline=self._entity_pipeline,
            relation_pipeline=self._relation_pipeline,
            chunk_size=config.get("chunk_size", 400),
        )

        # تنظیمات
        self._refresh_interval = config.get("refresh_interval", 60)
        self._sem = asyncio.Semaphore(config.get("max_concurrent", 5))
        self._last_refresh: dict[str, float] = {}
        self._active_domains: list[DataDomain] = config.get(
            "domains",
            [DataDomain.NEWS, DataDomain.KNOWLEDGE],
        )
        
        # متریک‌ها
        self._refresh_history: list[RefreshMetrics] = []
        self._ready = False

    # ══════════════════════════════════════════════════════════════
    # Lifecycle
    # ══════════════════════════════════════════════════════════════

    async def startup(self) -> None:
        """راه‌اندازی agent"""
        await self._kuzu.__aenter__()
        await self._weaviate.connect()
        self._ready = True
        logger.info("ImprovedKnowledgeRefreshAgent ready")

    async def shutdown(self) -> None:
        """خاموش کردن agent"""
        await self._kuzu.__aexit__(None, None, None)
        await self._weaviate.close()
        self._ready = False

    # ══════════════════════════════════════════════════════════════
    # Embedding Function
    # ══════════════════════════════════════════════════════════════

    def _make_embed_fn(self):
        """ساخت تابع embedding با مدیریت حافظه"""
        if self._use_model_manager and self._embed_wrapper:
            def _fn(text: str) -> list[float]:
                with self._embed_wrapper.use_model() as model:
                    result = model.embed_single(text)
                    return result["embedding"]
            return _fn
        elif self._embed_client:
            def _fn(text: str) -> list[float]:
                return self._embed_client.embed_single(text)["embedding"]
            return _fn
        else:
            def _fn(text: str) -> list[float]:
                logger.warning("No embed function available")
                return [0.0] * 1536
            return _fn

    # ══════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════

    async def refresh(self, state: AgentState) -> AgentState:
        """
        بروزرسانی دانش با استراتژی بهبود یافته
        """
        if not self._ready:
            raise RuntimeError("Call startup() first")

        if not state.refresh_needed:
            return state

        # بررسی throttle
        now = time()
        elapsed = now - self._last_refresh.get(state.query, 0)
        if elapsed < self._refresh_interval:
            logger.debug(
                "Throttled refresh for '%s' (%.0fs remaining)",
                state.query,
                self._refresh_interval - elapsed
            )
            return state

        self._last_refresh[state.query] = now
        logger.info("Starting improved knowledge refresh for: '%s'", state.query)

        start_time = time()
        metrics = RefreshMetrics()

        try:
            # ساخت task ها
            tasks = self._build_tasks(state)
            
            if not tasks:
                logger.warning("No refresh tasks built for '%s'", state.query)
                state.refresh_needed = False
                return state

            # اجرای موازی task ها
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # جمع‌آوری متریک‌ها
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Refresh task failed: %s", result)
                elif isinstance(result, dict):
                    metrics.total_items += result.get("items", 0)
                    metrics.chunks_ingested += result.get("chunks", 0)

            # بروزرسانی گراف معنایی
            graph_stats = await self._semantic_builder.get_graph_stats()
            logger.info("Graph stats after refresh: %s", graph_stats)

            # محاسبه duration
            metrics.duration_seconds = time() - start_time
            
            # ذخیره متریک‌ها
            self._refresh_history.append(metrics)
            if len(self._refresh_history) > 100:  # نگهداری آخرین 100 refresh
                self._refresh_history.pop(0)

            state.refresh_needed = False
            state.metadata["last_refresh_metrics"] = metrics.to_dict()

            logger.info(
                "Knowledge refresh completed in %.2fs | items=%d chunks=%d",
                metrics.duration_seconds,
                metrics.total_items,
                metrics.chunks_ingested
            )

        except Exception as e:
            logger.error("Refresh failed: %s", e, exc_info=True)
            state.current_step = FlowStep.ERROR
            state.error = str(e)

        return state

    async def incremental_refresh(
        self,
        state: AgentState,
        new_data: list,
    ) -> GraphUpdate:
        """
        بروزرسانی تدریجی با دانش جدید
        - فقط تغییرات را اعمال می‌کند
        - حل تعارض هوشمند
        """
        if not self._ready:
            raise RuntimeError("Call startup() first")

        logger.info("Starting incremental refresh with %d items", len(new_data))
        
        total_update = GraphUpdate()
        
        for item in new_data:
            # استخراج entities و relations
            chunks = self._chunk_pipeline._build_chunks([item])
            
            for chunk in chunks:
                # Entity extraction
                entities = await self._entity_pipeline.process_chunk(chunk)
                
                # Relation extraction
                relations = []
                if entities:
                    relations = self._relation_pipeline.extract(chunk, entities)
                
                # بروزرسانی تدریجی
                update = await self._semantic_builder.incremental_update(
                    new_entities=entities,
                    new_relations=relations,
                    chunk=chunk,
                )
                
                # جمع متریک‌ها
                total_update.nodes_added += update.nodes_added
                total_update.nodes_updated += update.nodes_updated
                total_update.edges_added += update.edges_added
                total_update.edges_updated += update.edges_updated
                total_update.conflicts_resolved += update.conflicts_resolved
        
        logger.info(
            "Incremental refresh done: +%d nodes, ~%d nodes, +%d edges, ~%d edges, %d conflicts",
            total_update.nodes_added,
            total_update.nodes_updated,
            total_update.edges_added,
            total_update.edges_updated,
            total_update.conflicts_resolved
        )
        
        return total_update

    # ══════════════════════════════════════════════════════════════
    # Task Building
    # ══════════════════════════════════════════════════════════════

    def _build_tasks(self, state: AgentState) -> list:
        """ساخت task های بروزرسانی"""
        tasks = []
        
        # استخراج ticker ها برای financial data
        tickers = self._extract_tickers(state.query)
        if tickers:
            tasks.append(self._run_refresh(
                self.data_manager.collect_financial,
                tickers=tickers,
            ))

        # domain-based collectors
        for domain in self._active_domains:
            collector_name = _DOMAIN_COLLECTORS.get(domain)
            if not collector_name:
                continue
            
            collector_fn = getattr(self.data_manager, collector_name, None)
            if collector_fn is None:
                logger.warning(
                    "Collector '%s' not found on DataCollectorManager",
                    collector_name
                )
                continue

            tasks.append(self._run_refresh(
                collector_fn,
                query=state.query,
            ))

        return tasks

    # ══════════════════════════════════════════════════════════════
    # Task Runner
    # ══════════════════════════════════════════════════════════════

    async def _run_refresh(self, fetch_fn, **kwargs) -> dict:
        """اجرای یک task بروزرسانی"""
        async with self._sem:
            try:
                # جمع‌آوری داده
                items = await asyncio.to_thread(fetch_fn, **kwargs)
                
                if not items:
                    logger.debug("%s returned no items", fetch_fn.__name__)
                    return {"items": 0, "chunks": 0}
                
                # Ingest با semantic graph builder
                chunks_ingested = await self._chunk_pipeline.ingest_items(items)
                
                logger.info(
                    "%s → %d items, %d chunks ingested",
                    fetch_fn.__name__,
                    len(items),
                    chunks_ingested
                )
                
                return {"items": len(items), "chunks": chunks_ingested}
                
            except Exception as e:
                logger.error("Task %s failed: %s", fetch_fn.__name__, e, exc_info=True)
                return {"items": 0, "chunks": 0}

    # ══════════════════════════════════════════════════════════════
    # Ticker Extraction (از نسخه قبل)
    # ══════════════════════════════════════════════════════════════

    _PERSIAN_TICKER_RE = re.compile(
        r'\b((?:خ|ف|و|ش|ت|ب|پ|ح|ق|ک|م|ز|ر|ن|ا|س|د|ل|ج|ع|غ|ی)'
        r'[\u0600-\u06FF]{1,5})\b'
    )
    
    _LATIN_TICKER_RE = re.compile(r'\b([A-Z]{2,5})\b')
    
    _TICKER_BLACKLIST = {
        "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT",
        "ARE", "WAS", "HAS", "NOT", "BUT", "CAN", "WILL",
    }
    
    _PERSIAN_STOPWORDS = {
        'من', 'تو', 'او', 'ما', 'شما', 'آنها', 'این', 'آن',
        'به', 'از', 'در', 'با', 'بر', 'که', 'است', 'بود',
    }
    
    _VALID_PERSIAN_TICKER_STARTS = frozenset('خفوشتبپحقکمزرناسدلجعغیثصضطظء')
    
    def _extract_tickers(self, query: str) -> list[str]:
        """استخراج ticker های بورسی از query"""
        result = []
        seen = set()
        
        # لاتین
        for match in self._LATIN_TICKER_RE.finditer(query):
            t = match.group(1)
            if (
                t not in seen
                and t not in self._TICKER_BLACKLIST
                and self._is_valid_latin_ticker(t)
            ):
                seen.add(t)
                result.append(t)
        
        # فارسی
        for match in self._PERSIAN_TICKER_RE.finditer(query):
            t = match.group(1)
            if t not in seen and self._is_valid_persian_ticker(t):
                seen.add(t)
                result.append(t)
        
        return result
    
    def _is_valid_latin_ticker(self, t: str) -> bool:
        """اعتبارسنجی ticker لاتین"""
        return (
            t.isupper()
            and 2 <= len(t) <= 5
            and len(set(t)) > 1
            and t not in self._TICKER_BLACKLIST
        )
    
    def _is_valid_persian_ticker(self, t: str) -> bool:
        """اعتبارسنجی ticker فارسی"""
        return (
            3 <= len(t) <= 6
            and t not in self._PERSIAN_STOPWORDS
            and t[0] in self._VALID_PERSIAN_TICKER_STARTS
            and len(set(t)) > 1
            and not any(c.isdigit() for c in t)
        )

    # ══════════════════════════════════════════════════════════════
    # Metrics & Monitoring
    # ══════════════════════════════════════════════════════════════

    def get_refresh_history(self, limit: int = 10) -> list[dict]:
        """دریافت تاریخچه بروزرسانی‌ها"""
        return [m.to_dict() for m in self._refresh_history[-limit:]]

    async def get_graph_quality_metrics(self) -> dict:
        """محاسبه متریک‌های کیفیت گراف"""
        stats = await self._semantic_builder.get_graph_stats()
        
        # محاسبه متریک‌های اضافی
        entity_count = stats.get("entity_count", 0)
        relation_count = stats.get("relation_count", 0)
        
        density = 0.0
        if entity_count > 1:
            max_edges = entity_count * (entity_count - 1)
            density = relation_count / max_edges if max_edges > 0 else 0.0
        
        avg_degree = (2 * relation_count / entity_count) if entity_count > 0 else 0.0
        
        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "graph_density": round(density, 4),
            "average_degree": round(avg_degree, 2),
            "refresh_count": len(self._refresh_history),
        }

