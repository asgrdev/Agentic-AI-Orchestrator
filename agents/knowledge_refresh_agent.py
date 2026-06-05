# agents/knowledge_refresh_agent.py
from __future__ import annotations

import asyncio
import logging
import re
from time import time

from agents.state import AgentState, FlowStep
from external_sources.data_collector.manager import DataCollectorManager
from external_sources.data_collector.models import DataDomain
from ingestion.chunk_ingestion_pipeline import ChunkIngestionPipeline
from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager
from vector_store.weaviate_client import WeaviateStore
from ingestion.embedding_generator import Qwen3EmbeddingClient
from ingestion.graph_builder import GraphBuilder
from ingestion.relation_pipeline import RelationPipeline
from ingestion.EntityExtractionPipeline import EntityExtractionPipeline

logger = logging.getLogger(__name__)

# ticker های رایج بازار ایران + بین‌الملل
_TICKER_RE = re.compile(
    r'\b([A-Z]{2,5}|[\u0600-\u06FF]{2,6})\b'
)

# کلمات blacklist که ticker نیستند
_TICKER_BLACKLIST = {
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT",
    "ARE", "WAS", "HAS", "NOT", "BUT", "CAN", "WILL",
}

# نگاشت domain → متد collect در DataCollectorManager
_DOMAIN_COLLECTORS = {
    DataDomain.KNOWLEDGE:   "collect_knowledge",
    DataDomain.SCIENTIFIC:  "collect_scientific",
    DataDomain.NEWS:        "collect_political_news",
    DataDomain.SOCIAL:      "collect_social",
}


class KnowledgeRefreshAgent:

    def __init__(self, config: dict):

        if "embed_client_factory" in config:
            embed_client = config["embed_client_factory"]()
        else:
            embed_client = config["embed_client"]

        self._embed_client: Qwen3EmbeddingClient = embed_client

        self.data_manager: DataCollectorManager = config.get(
            "data_manager", DataCollectorManager()
        )

        kuzu     = AsyncKuzuManager(config["kuzu_path"])
        wv       = config.get("weaviate", {})
        weaviate = WeaviateStore(
            collection_name=config.get("collection_name", "Chunks"),
            mode=wv.get("mode", "docker"),   # docker by default
            host=wv.get("host", "localhost"),
            port=wv.get("port", 8080),
            grpc_port=wv.get("grpc_port", 50051),
            cloud_url=wv.get("cloud_url", ""),
            api_key=wv.get("api_key", ""),
            vector_dim=self._embed_client.embedding_dim,
        )

        self._pipeline = ChunkIngestionPipeline(
            weaviate=weaviate,
            kuzu=kuzu,
            embed_fn=self._make_embed_fn(),
            entity_pipeline=config.get("entity_pipeline"),
            chunk_size=config.get("chunk_size", 400),
        )
        self._kuzu     = kuzu
        self._weaviate = weaviate

        self._refresh_interval = config.get("refresh_interval", 60)
        self._sem              = asyncio.Semaphore(config.get("max_concurrent", 5))
        self._last_refresh:  dict[str, float] = {}
        self._active_domains: list[DataDomain] = config.get(
            "domains",
            [DataDomain.NEWS, DataDomain.KNOWLEDGE],
        )
        self._ready = False


    async def _initialize_pipelines(self) -> None:
        """Initialize ingestion pipelines with entity/relation extraction."""
        try:
            # Embedding client
            embed_client = Qwen3EmbeddingClient(model_path="/Users/dbk/Desktop/agentic-graph-RAG/models/Qwen3-Embedding-0.6B")
            
            # Entity extraction pipeline
            entity_config = {
                "ner_model": "dslim/bert-base-NER",
                "entity_linking_endpoint": self.config.get("entity_linking_endpoint"),
                "max_workers": 4,
            }
            entity_pipeline = EntityExtractionPipeline(entity_config)
            
            # Relation pipeline
            relation_config = {
                "llm_endpoint": self.config.get("llm_endpoint"),
                "model_name": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            }
            relation_pipeline = RelationPipeline(relation_config)
            
            # Graph builder
            graph_builder = GraphBuilder(self.kuzu)
            
            # Chunk ingestion pipeline
            self.chunk_pipeline = ChunkIngestionPipeline(
                weaviate=self.weaviate,
                kuzu=self.kuzu,
                embed_fn=lambda text: embed_client.embed([text])[0],
                entity_pipeline=entity_pipeline,
                relation_pipeline=relation_pipeline,
                graph_builder=graph_builder,
                chunk_size=400,
            )
            
            logger.info("Pipelines initialized with entity/relation extraction")
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise


    # ── Lifecycle ──────────────────────────────────────────────────────

    async def startup(self) -> None:
        await self._kuzu.__aenter__()
        await self._weaviate.connect()
        self._ready = True
        logger.info("KnowledgeRefreshAgent ready")

    async def shutdown(self) -> None:
        await self._kuzu.__aexit__(None, None, None)
        await self._weaviate.close()

    # ── embed_fn برای pipeline ─────────────────────────────────────────

    def _make_embed_fn(self):
        """
        ChunkIngestionPipeline انتظار Callable[[str], list[float]] دارد.
        از embed_single استفاده می‌کنیم چون pipeline خودش async_gather دارد.
        """
        def _fn(text: str) -> list[float]:
            return self._embed_client.embed_single(text)["embedding"]
        return _fn

    # ── Public API ─────────────────────────────────────────────────────

    async def refresh(self, state: AgentState) -> AgentState:
        if not self._ready:
            raise RuntimeError("Call startup() first")

        if not state.refresh_needed:
            return state

        now     = time()
        elapsed = now - self._last_refresh.get(state.query, 0)
        if elapsed < self._refresh_interval:
            logger.debug("Throttled refresh for '%s' (%.0fs remaining)",
                         state.query, self._refresh_interval - elapsed)
            return state

        self._last_refresh[state.query] = now
        logger.info("Refreshing knowledge for: '%s'", state.query)

        try:
            tasks = self._build_tasks(state)
            logger.warning(">>>t1> '%s' <", tasks, exc_info=True)

            if not tasks:
                logger.warning("No refresh tasks built for '%s'", state.query)
                state.refresh_needed = False
                return state
            
            logger.warning(">>>t2> '%s' <", tasks, exc_info=True)

            await asyncio.gather(*tasks, return_exceptions=True)
            state.refresh_needed = False

        except Exception as e:
            logger.error("Refresh failed: %s", e, exc_info=True)
            state.current_step = FlowStep.ERROR
        
        logger.warning(">>>t3> '%s' --'%s'<", tasks, state.refresh_needed, exc_info=True)

        return state

    # ── Task Builder ───────────────────────────────────────────────────

    def _build_tasks(self, state: AgentState) -> list:
        tasks = []
        tickers = self._extract_tickers(state.query)

        # financial فقط اگر ticker پیدا شد
        if tickers:
            tasks.append(self._run_refresh(
                self.data_manager.collect_financial,
                tickers=tickers,
            ))

        # domain-based collectors
        for domain in self._active_domains:
            logger.warning(">>>Domains> '%s' <-", domain)
            collector_name = _DOMAIN_COLLECTORS.get(domain)
            if not collector_name:
                continue
            logger.warning(">>>> '%s' -- '%s' <", collector_name, self.data_manager)
            collector_fn = getattr(self.data_manager, collector_name, None)
            if collector_fn is None:
                logger.warning("Collector '%s' not found on DataCollectorManager", collector_name)
                continue

            tasks.append(self._run_refresh(
                collector_fn,
                query=state.query,
            ))

        return tasks

    # ── Runner ─────────────────────────────────────────────────────────

    async def _run_refresh(self, fetch_fn, **kwargs) -> None:
        logger.info("%s <<< _run_refresh", fetch_fn)
        logger.info("%s <<< _run_refresh", self._sem)
        async with self._sem:
            try:
                logger.info("%s <<< inThe_run_refresh",fetch_fn )
                print(f"IN>>>>${kwargs}")
                items = await asyncio.to_thread(fetch_fn, **kwargs)
                if not items:
                    logger.debug("%s returned no items", fetch_fn.__name__)
                    return
                ingested = await self._pipeline.ingest_items(items)
                logger.info("%s → %d chunks ingested", fetch_fn.__name__, ingested)
            except Exception as e:
                logger.error("Task %s failed: %s", fetch_fn.__name__, e, exc_info=True)

    # ── Ticker Extraction ──────────────────────────────────────────────
# نمادهای بورسی ایران الگوی مشخصی دارند
    _PERSIAN_TICKER_RE = re.compile(
        r'\b((?:خ|ف|و|ش|ت|ب|پ|ح|ق|ک|م|ز|ر|ن|ا|س|د|ل|ج|ع|غ|ی)'  # حرف اول خاص
        r'[\u0600-\u06FF]{1,5})\b'                                    # ۲ تا ۶ کاراکتر کل
    )
    
    # لاتین فقط ۲ تا ۵ حرف بزرگ - بدون عدد داخلش
    _LATIN_TICKER_RE = re.compile(
        r'\b([A-Z]{2,5})\b'
    )
        
    _PERSIAN_STOPWORDS = {
        # ضمایر
        'من', 'تو', 'او', 'ما', 'شما', 'آنها', 'آن', 'این', 'وی',
        # حروف اضافه
        'به', 'از', 'در', 'با', 'بر', 'تا', 'برای', 'علیه', 'میان',
        'بین', 'روی', 'زیر', 'بالا', 'پایین', 'کنار', 'نزد', 'نزدیک',
        #接続詞 /接続詞
        'که', 'اگر', 'اما', 'ولی', 'یا', 'پس', 'چون', 'زیرا', 'چرا',
        'بلکه', 'هرچند', 'گرچه', 'وقتی', 'هنگامی',
        # افعال رایج
        'است', 'بود', 'شد', 'کرد', 'هست', 'نیست', 'بده', 'بگو',
        'دارم', 'دارد', 'داری', 'دارند', 'داشت', 'داشتم', 'میشه',
        'میکنه', 'میکنم', 'میکنی', 'بکن', 'بکنم', 'کنم', 'کنی',
        'میخوام', 'میخوای', 'میخواد', 'بخوام', 'خواست',
        # صفات/قیدهای رایج
        'خوب', 'بد', 'زیاد', 'کم', 'خیلی', 'خیلیی', 'خوبه', 'بده',
        'بزرگ', 'کوچک', 'جدید', 'قدیم', 'دور', 'نزدیک', 'سریع',
        # اسامی رایج (غیر بورسی)
        'ایران', 'تهران', 'بازار', 'قیمت', 'نرخ', 'سود', 'زیان',
        'فارسی', 'انگلیسی', 'عربی', 'وضعیت', 'گزارش', 'اطلاعات',
        'جواب', 'سوال', 'نیاز', 'مشکل', 'راه', 'حل', 'کمک',
        'گذشته', 'آینده', 'امروز', 'دیروز', 'فردا', 'الان', 'الآن',
        'هفته', 'ماه', 'سال', 'روز', 'ساعت', 'دقیقه',
        'بحال', 'حالا', 'همین', 'همان', 'هنوز', 'دیگر', 'دیگه',
        # اعداد فارسی
        'یک', 'دو', 'سه', 'چهار', 'پنج', 'شش', 'هفت', 'هشت', 'نه', 'ده',
        # متفرقه
        'هم', 'هر', 'همه', 'چه', 'چی', 'کی', 'کجا', 'چند', 'چطور',
        'چطوره', 'اینجا', 'آنجا', 'اونجا', 'اینقدر', 'انقدر',
    }
    
    # حروف اول معتبر برای نماد بورسی فارسی
    _VALID_PERSIAN_TICKER_STARTS = frozenset(
        'خفوشتبپحقکمزرناسدلجعغیثصضطظء'
    )
    
    # حداقل طول معنادار
    _MIN_PERSIAN_TICKER_LEN = 3
    _MAX_PERSIAN_TICKER_LEN = 6
    
    def _extract_tickers(self, query: str) -> list[str]:
       result = []
       seen = set()
     
       # --- لاتین ---
       for match in self._LATIN_TICKER_RE.finditer(query):
           t = match.group(1)
           if (
               t not in seen
               and t not in _TICKER_BLACKLIST
               and self._is_valid_latin_ticker(t)
           ):
               seen.add(t)
               result.append(t)
     
       # --- فارسی ---
       for match in self._PERSIAN_TICKER_RE.finditer(query):
           t = match.group(1)
           if (
               t not in seen
               and self._is_valid_persian_ticker(t)
           ):
               seen.add(t)
               result.append(t)
     
       if result:
           logger.debug("Extracted tickers: %s", result)
     
       return result
   
   
    def _is_valid_latin_ticker(self, t: str) -> bool:
        """
        فیلتر چندلایه برای نمادهای لاتین
        """
        # همه حروف بزرگ باشد
        if not t.isupper():
            return False
        # طول معتبر
        if not (2 <= len(t) <= 5):
            return False
        # نباید فقط یک حرف تکراری باشد (AAA, BBBB)
        if len(set(t)) == 1:
            return False
        # در blacklist نباشد
        if t in _TICKER_BLACKLIST:
            return False
        return True
    
    
    def _is_valid_persian_ticker(self, t: str) -> bool:
        """
        فیلتر چندلایه برای نمادهای فارسی
        """
        # طول معتبر
        if not (self._MIN_PERSIAN_TICKER_LEN <= len(t) <= self._MAX_PERSIAN_TICKER_LEN):
            return False
        # در stopwords نباشد
        if t in self._PERSIAN_STOPWORDS:
            return False
        # حرف اول باید از حروف معتبر نماد بورسی باشد
        if t[0] not in self._VALID_PERSIAN_TICKER_STARTS:
            return False
        # نباید فقط یک حرف تکراری باشد
        if len(set(t)) == 1:
            return False
        # نباید عدد داشته باشد
        if any(c.isdigit() for c in t):
            return False
        return True
