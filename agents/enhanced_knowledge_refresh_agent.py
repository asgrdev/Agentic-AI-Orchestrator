"""
Enhanced Knowledge Refresh Agent - بروزرسانی هوشمند دانش با گراف معنایی

این ماژول فرآیند بروزرسانی دانش را با استفاده از SemanticGraphBuilder بهبود می‌دهد:
- یکپارچگی کامل با گراف معنایی
- بروزرسانی تدریجی و هوشمند
- حل تعارض خودکار
- نسخه‌بندی و تاریخچه
- اعتبارسنجی کیفیت گراف
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import TYPE_CHECKING, Any, Optional

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

if TYPE_CHECKING:
    from ingestion.datasets.entities import Entity
    from ingestion.datasets.relations import Relation

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
    quality_score: float = 0.0
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    errors: list[str] = field(default_factory=list)


@dataclass
class RefreshConfig:
    """تنظیمات بروزرسانی دانش"""
    refresh_interval: int = 60  # ثانیه
    max_concurrent: int = 5
    enable_inference: bool = True
    enable_versioning: bool = True
    confidence_threshold: float = 0.5
    quality_threshold: float = 0.6
    active_domains: list[DataDomain] = field(default_factory=lambda: [
        DataDomain.NEWS,
        DataDomain.KNOWLEDGE,
    ])


class EnhancedKnowledgeRefreshAgent:
    """
    Agent بروزرسانی دانش با قابلیت‌های پیشرفته:
    - یکپارچگی با SemanticGraphBuilder
    - بروزرسانی تدریجی و هوشمند
    - اعتبارسنجی کیفیت
    - متریک‌های جامع
    """

    def __init__(self, config: dict) -> None:
        # Embedding client
        if "embed_client_factory" in config:
            self._embed_client = config["embed_client_factory"]()
        else:
            self._embed_client = config["embed_client"]

        # Data collector
        self.data_manager = config.get("data_manager", DataCollectorManager())

        # Kuzu graph manager
        self._kuzu = AsyncKuzuManager(config["kuzu_path"])

        # Weaviate vector store
        wv_config = config.get("weaviate", {})
        self._weaviate = WeaviateStore(
            collection_name=config.get("collection_name", "Chunks"),
            mode=wv_config.get("mode", "docker"),
            host=wv_config.get("host", "localhost"),
            port=wv_config.get("port", 8080),
            grpc_port=wv_config.get("grpc_port", 50051),
            cloud_url=wv_config.get("cloud_url", ""),
            api_key=wv_config.get("api_key", ""),
            vector_dim=self._embed_client.embedding_dim,
        )

        # Refresh configuration
        self._config = RefreshConfig(
            refresh_interval=config.get("refresh_interval", 60),
            max_concurrent=config.get("max_concurrent", 5),
            enable_inference=config.get("enable_inference", True),
            enable_versioning=config.get("enable_versioning", True),
            confidence_threshold=config.get("confidence_threshold", 0.5),
            quality_threshold=config.get("quality_threshold", 0.6),
            active_domains=config.get("domains", [DataDomain.NEWS, DataDomain.KNOWLEDGE]),
        )

        # Semantic graph builder
        self._semantic_builder: Optional[SemanticGraphBuilder] = None

        # Pipelines (initialized in startup)
        self._entity_pipeline: Optional[EntityExtractionPipeline] = None
        self._relation_pipeline: Optional[RelationPipeline] = None
        self._chunk_pipeline: Optional[ChunkIngestionPipeline] = None

        # State management
        self._sem = asyncio.Semaphore(self._config.max_concurrent)
        self._last_refresh: dict[str, float] = {}
        self._refresh_history: list[RefreshMetrics] = []
        self._ready = False

    # ══════════════════════════════════════════════════════════════
    # Lifecycle Management
    # ══════════════════════════════════════════════════════════════

    async def startup(self) -> None:
        """راه‌اندازی agent و تمام وابستگی‌ها"""
        try:
            # Connect to databases
            await self._kuzu.__aenter__()
            await self._weaviate.connect()

            # Initialize semantic graph builder
            self._semantic_builder = SemanticGraphBuilder(
                graph_manager=self._kuzu,
                enable_inference=self._config.enable_inference,
                enable_versioning=self._config.enable_versioning,
                confidence_threshold=self._config.confidence_threshold,
            )

            # Initialize pipelines
            await self._initialize_pipelines()

            self._ready = True
            logger.info("✅ EnhancedKnowledgeRefreshAgent ready")
            logger.info("   - Semantic graph: enabled")
            logger.info("   - Inference: %s", self._config.enable_inference)
            logger.info("   - Versioning: %s", self._config.enable_versioning)
            logger.info("   - Active domains: %s", [d.value for d in self._config.active_domains])

        except Exception as e:
            logger.error("❌ Startup failed: %s", e, exc_info=True)
            raise

    async def shutdown(self) -> None:
        """خاموش کردن agent و آزادسازی منابع"""
        try:
            await self._kuzu.__aexit__(None, None, None)
            await self._weaviate.close()
            logger.info("✅ EnhancedKnowledgeRefreshAgent shutdown complete")
        except Exception as e:
            logger.error("❌ Shutdown error: %s", e)

    async def _initialize_pipelines(self) -> None:
        """راه‌اندازی pipeline های استخراج و پردازش"""
        try:
            # Entity extraction pipeline
            entity_config = {
                "ner_model": "dslim/bert-base-NER",
                "max_workers": 4,
            }
            self._entity_pipeline = EntityExtractionPipeline(entity_config)

            # Relation extraction pipeline
            relation_config = {
                "model_name": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            }
            self._relation_pipeline = RelationPipeline(relation_config)

            # Chunk ingestion pipeline
            self._chunk_pipeline = ChunkIngestionPipeline(
                weaviate=self._weaviate,
                kuzu=self._kuzu,
                embed_fn=self._make_embed_fn(),
                entity_pipeline=self._entity_pipeline,
                relation_pipeline=self._relation_pipeline,
                graph_builder=None,  # استفاده از semantic builder
                chunk_size=400,
            )

            logger.info("✅ Pipelines initialized")

        except Exception as e:
            logger.error("❌ Pipeline initialization failed: %s", e)
            raise

    def _make_embed_fn(self):
        """تابع embedding برای pipeline"""
        def _fn(text: str) -> list[float]:
            return self._embed_client.embed_single(text)["embedding"]
        return _fn

    # ══════════════════════════════════════════════════════════════
    # Public API - Knowledge Refresh
    # ══════════════════════════════════════════════════════════════

    async def refresh(self, state: AgentState) -> AgentState:
        """
        بروزرسانی دانش با استفاده از گراف معنایی
        
        Args:
            state: وضعیت فعلی agent
            
        Returns:
            state بروزرسانی شده با متریک‌ها
        """
        if not self._ready:
            raise RuntimeError("Call startup() first")

        if not state.refresh_needed:
            return state

        # Throttling check
        now = time()
        elapsed = now - self._last_refresh.get(state.query, 0)
        if elapsed < self._config.refresh_interval:
            logger.debug(
                "⏳ Throttled refresh for '%s' (%.0fs remaining)",
                state.query,
                self._config.refresh_interval - elapsed
            )
            return state

        self._last_refresh[state.query] = now
        start_time = time()

        logger.info("🔄 Starting knowledge refresh for: '%s'", state.query)

        metrics = RefreshMetrics()

        try:
            # 1. جمع‌آوری داده‌های جدید
            items = await self._collect_data(state.query, metrics)
            
            if not items:
                logger.warning("⚠️  No new data collected for '%s'", state.query)
                state.refresh_needed = False
                return state

            # 2. پردازش و استخراج entities/relations
            processed_data = await self._process_items(items, metrics)

            # 3. بروزرسانی گراف معنایی
            graph_updates = await self._update_semantic_graph(processed_data, metrics)

            # 4. اعتبارسنجی کیفیت
            quality_score = await self._validate_graph_quality()
            metrics.quality_score = quality_score

            # 5. ذخیره متریک‌ها
            metrics.duration_seconds = time() - start_time
            self._refresh_history.append(metrics)

            # 6. بروزرسانی state
            state.refresh_needed = False
            state.metadata["refresh_metrics"] = {
                "chunks_ingested": metrics.chunks_ingested,
                "nodes_added": metrics.nodes_added,
                "edges_added": metrics.edges_added,
                "inferred_relations": metrics.inferred_relations,
                "quality_score": metrics.quality_score,
                "duration": metrics.duration_seconds,
            }

            logger.info(
                "✅ Knowledge refresh complete: "
                "+%d nodes, +%d edges, %d inferred, quality=%.2f (%.1fs)",
                metrics.nodes_added,
                metrics.edges_added,
                metrics.inferred_relations,
                metrics.quality_score,
                metrics.duration_seconds
            )

        except Exception as e:
            logger.error("❌ Refresh failed: %s", e, exc_info=True)
            metrics.errors.append(str(e))
            state.current_step = FlowStep.ERROR
            state.metadata["refresh_error"] = str(e)

        return state

    # ══════════════════════════════════════════════════════════════
    # Data Collection
    # ══════════════════════════════════════════════════════════════

    async def _collect_data(
        self,
        query: str,
        metrics: RefreshMetrics
    ) -> list[dict]:
        """جمع‌آوری داده از منابع مختلف"""
        tasks = []
        tickers = self._extract_tickers(query)

        # Financial data (if tickers found)
        if tickers:
            tasks.append(self._collect_financial(tickers))

        # Domain-based collectors
        for domain in self._config.active_domains:
            collector_fn = self._get_domain_collector(domain)
            if collector_fn:
                tasks.append(self._collect_domain(collector_fn, query))

        if not tasks:
            return []

        # Run all collectors concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate items
        all_items = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Collector failed: %s", result)
                metrics.errors.append(str(result))
            elif isinstance(result, list):
                all_items.extend(result)

        metrics.total_items = len(all_items)
        logger.info("📥 Collected %d items from %d sources", len(all_items), len(tasks))

        return all_items

    async def _collect_financial(self, tickers: list[str]) -> list[dict]:
        """جمع‌آوری داده‌های مالی"""
        async with self._sem:
            try:
                items = await asyncio.to_thread(
                    self.data_manager.collect_financial,
                    tickers=tickers
                )
                return items or []
            except Exception as e:
                logger.error("Financial collection failed: %s", e)
                return []

    async def _collect_domain(self, collector_fn, query: str) -> list[dict]:
        """جمع‌آوری داده از یک domain"""
        async with self._sem:
            try:
                items = await asyncio.to_thread(collector_fn, query=query)
                return items or []
            except Exception as e:
                logger.error("Domain collection failed: %s", e)
                return []

    def _get_domain_collector(self, domain: DataDomain):
        """دریافت تابع collector برای domain"""
        domain_map = {
            DataDomain.KNOWLEDGE: "collect_knowledge",
            DataDomain.SCIENTIFIC: "collect_scientific",
            DataDomain.NEWS: "collect_political_news",
            DataDomain.SOCIAL: "collect_social",
        }
        collector_name = domain_map.get(domain)
        if collector_name:
            return getattr(self.data_manager, collector_name, None)
        return None

    # ══════════════════════════════════════════════════════════════
    # Data Processing
    # ══════════════════════════════════════════════════════════════

    async def _process_items(
        self,
        items: list[dict],
        metrics: RefreshMetrics
    ) -> list[dict]:
        """پردازش items و استخراج entities/relations"""
        processed = []

        for item in items:
            try:
                # Extract text content
                text = item.get("content", "") or item.get("text", "")
                if not text:
                    continue

                # Create chunk
                chunk = {
                    "id": item.get("id", f"chunk_{hash(text)}"),
                    "title": item.get("title", ""),
                    "text": text,
                    "source": item.get("source", ""),
                    "timestamp": item.get("timestamp", datetime.now().isoformat()),
                }

                # Extract entities
                entities = await self._entity_pipeline.extract(text)

                # Extract relations
                relations = await self._relation_pipeline.extract(text, entities)

                processed.append({
                    "chunk": chunk,
                    "entities": entities,
                    "relations": relations,
                })

                metrics.chunks_ingested += 1

            except Exception as e:
                logger.error("Item processing failed: %s", e)
                metrics.errors.append(f"Processing: {str(e)}")

        logger.info("⚙️  Processed %d/%d items", len(processed), len(items))
        return processed

    # ══════════════════════════════════════════════════════════════
    # Semantic Graph Update
    # ══════════════════════════════════════════════════════════════

    async def _update_semantic_graph(
        self,
        processed_data: list[dict],
        metrics: RefreshMetrics
    ) -> list[GraphUpdate]:
        """بروزرسانی گراف معنایی با داده‌های جدید"""
        updates = []

        for data in processed_data:
            try:
                # Build semantic graph for this chunk
                update = await self._semantic_builder.build_semantic_graph(
                    chunk=data["chunk"],
                    entities=data["entities"],
                    relations=data["relations"],
                )

                updates.append(update)

                # Aggregate metrics
                metrics.nodes_added += update.nodes_added
                metrics.nodes_updated += update.nodes_updated
                metrics.edges_added += update.edges_added
                metrics.edges_updated += update.edges_updated
                metrics.inferred_relations += update.inferred_relations
                metrics.conflicts_resolved += update.conflicts_resolved

            except Exception as e:
                logger.error("Graph update failed: %s", e)
                metrics.errors.append(f"Graph: {str(e)}")

        logger.info(
            "📊 Graph updated: +%d nodes, ~%d nodes, +%d edges, %d inferred",
            metrics.nodes_added,
            metrics.nodes_updated,
            metrics.edges_added,
            metrics.inferred_relations
        )

        return updates

    # ══════════════════════════════════════════════════════════════
    # Quality Validation
    # ══════════════════════════════════════════════════════════════

    async def _validate_graph_quality(self) -> float:
        """اعتبارسنجی کیفیت گراف"""
        try:
            stats = await self._semantic_builder.get_graph_stats()
            
            entity_count = stats.get("entity_count", 0)
            relation_count = stats.get("relation_count", 0)

            if entity_count == 0:
                return 0.0

            # محاسبه نسبت روابط به entities
            relation_ratio = relation_count / entity_count if entity_count > 0 else 0

            # Quality score based on graph density
            # Ideal ratio: 2-5 relations per entity
            if relation_ratio < 1:
                quality = relation_ratio * 0.5  # Low connectivity
            elif relation_ratio <= 3:
                quality = 0.5 + (relation_ratio - 1) * 0.25  # Good range
            elif relation_ratio <= 5:
                quality = 1.0  # Optimal
            else:
                quality = max(0.8, 1.0 - (relation_ratio - 5) * 0.05)  # Too dense

            return min(1.0, quality)

        except Exception as e:
            logger.error("Quality validation failed: %s", e)
            return 0.0

    # ══════════════════════════════════════════════════════════════
    # Ticker Extraction (from original agent)
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

    def _extract_tickers(self, query: str) -> list[str]:
        """استخراج نمادهای بورسی از query"""
        result = []
        seen = set()

        # Latin tickers
        for match in self._LATIN_TICKER_RE.finditer(query):
            t = match.group(1)
            if t not in seen and t not in self._TICKER_BLACKLIST:
                seen.add(t)
                result.append(t)

        # Persian tickers
        for match in self._PERSIAN_TICKER_RE.finditer(query):
            t = match.group(1)
            if t not in seen:
                seen.add(t)
                result.append(t)

        return result

    # ══════════════════════════════════════════════════════════════
    # Metrics & Monitoring
    # ══════════════════════════════════════════════════════════════

    def get_refresh_history(self, limit: int = 10) -> list[dict]:
        """دریافت تاریخچه بروزرسانی‌ها"""
        history = self._refresh_history[-limit:]
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "total_items": m.total_items,
                "chunks_ingested": m.chunks_ingested,
                "nodes_added": m.nodes_added,
                "edges_added": m.edges_added,
                "inferred_relations": m.inferred_relations,
                "quality_score": m.quality_score,
                "duration": m.duration_seconds,
                "errors": len(m.errors),
            }
            for m in history
        ]

    async def get_graph_stats(self) -> dict[str, Any]:
        """آمار فعلی گراف"""
        if self._semantic_builder:
            return await self._semantic_builder.get_graph_stats()
        return {}

    def clear_cache(self) -> None:
        """پاک کردن کش"""
        if self._semantic_builder:
            self._semantic_builder.clear_cache()
        self._last_refresh.clear()


