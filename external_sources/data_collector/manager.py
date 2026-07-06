import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Callable
import json
import re
import time

from external_sources.data_collector.models import (
    CollectedItem, CollectionResult,
    SearchQuery, DataSource, DataDomain
)
from external_sources.data_collector.helper import DataHelper
from external_sources.data_collector.exceptions import CollectorNotFoundError
from external_sources.data_collector.collectors.knowledge  import KnowledgeCollector
from external_sources.data_collector.collectors.scientific import ScientificCollector
from external_sources.data_collector.collectors.social     import SocialCollector
from external_sources.data_collector.collectors.financial  import FinancialCollector

console = Console()


class DataCollectorManager:
    """
    مرکز فرمان یکپارچه برای همه collectors

    نمونه استفاده:
        manager = DataCollectorManager(
            reddit_client_id="...",
            reddit_client_secret="...",
            fred_api_key="...",
        )
        results = manager.search_all("artificial intelligence", max_per_source=5)
    """

    def __init__(
        self,
        language:             str = "en",
        reddit_client_id:     str | None = None,
        reddit_client_secret: str | None = None,
        fred_api_key:         str | None = None,
        max_workers:          int = 4,
    ):
        self.max_workers = max_workers

        # ─── راه‌اندازی collectors ─────────────────────────────────────
        self.knowledge  = KnowledgeCollector(language=language)
        self.scientific = ScientificCollector()
        self.social     = SocialCollector(
            reddit_client_id=reddit_client_id,
            reddit_client_secret=reddit_client_secret,
        )
        self.financial  = FinancialCollector(fred_api_key=fred_api_key)

        # ─── نگاشت source → تابع جستجو ─────────────────────────────────
        self._search_registry: dict[DataSource, Callable] = {
            DataSource.WIKIPEDIA:        self.knowledge.search_wikipedia,
            DataSource.ARXIV:            self.scientific.search_arxiv,
            DataSource.OPENALEX:         self.scientific.search_openalex,
            DataSource.SEMANTIC_SCHOLAR: self.scientific.search_semantic_scholar,
            DataSource.REDDIT:           self.social.search_reddit,
        }

        logger.info("✅ DataCollectorManager راه‌اندازی شد")

    # ─── جستجوی تک منبع ─────────────────────────────────────────────────

    def search(
        self,
        query: str,
        source: DataSource,
        max_results: int = 10,
        **kwargs
    ) -> CollectionResult:
        if source not in self._search_registry:
            raise CollectorNotFoundError(f"Source '{source}' not supported for search")

        func = self._search_registry[source]
        logger.info(f"🔍 [{source}] searching: '{query}'")
        result = func(query=query, max_results=max_results, **kwargs)
        DataHelper.print_result_summary(result)
        return result

    # ─── جستجوی موازی همه منابع ─────────────────────────────────────────

    def search_all(
    self,
    query: str,
    sources: list[DataSource] | None = None,
    max_per_source: int = 5,
    domain_filter: DataDomain | None = None,
    timeout_per_source: float = 30.0,  # ← اضافه شد
) -> dict[DataSource, CollectionResult]:
        logger.info("🔍 search_all started | query={!r} | sources={}", query, sources)
        if sources is None:
            sources = list(self._search_registry.keys())
    
        results: dict[DataSource, CollectionResult] = {}
    
        console.print(f"\n[bold cyan]🌐 جستجوی موازی: '{query}'[/bold cyan]")
        console.print(f"منابع: {[s.value for s in sources]}\n")
    
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
    
            tasks_map = {
                source: progress.add_task(f"[yellow]{source.value}...", total=None)
                for source in sources
            }
    
            def _run(source: DataSource):
                try:
                    result = self._search_registry[source](
                        query=query, max_results=max_per_source
                    )
                    progress.update(tasks_map[source], description=f"[green]✅ {source.value}")
                    return source, result
                except Exception as e:
                    progress.update(tasks_map[source], description=f"[red]❌ {source.value}")
                    return source, CollectionResult(
                        success=False,
                        source=source,
                        total=0,
                        items=[],
                        errors=[str(e)]
                    )
    
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = {ex.submit(_run, src): src for src in sources}
                for future in as_completed(futures, timeout=timeout_per_source * len(sources)):
                    src = futures[future]
                    try:
                        source, result = future.result(timeout=timeout_per_source)
                    except TimeoutError:
                        progress.update(tasks_map[src], description=f"[red]⏱ {src.value} (timeout)")
                        logger.warning(f"⏱ [{src.value}] timeout after {timeout_per_source}s")
                        results[src] = CollectionResult(
                            success=False,
                            source=src,
                            total=0,
                            items=[],
                            errors=[f"Timeout after {timeout_per_source}s"]
                        )
                        continue
                    results[source] = result
    
        self._print_summary_table(results)
        return results


    # ─── جمع‌آوری تخصصی ─────────────────────────────────────────────────

    def collect_scientific(
        self,
        query: str,
        max_results: int = 10
    ) -> list[CollectedItem]:
        """جمع‌آوری علمی از همه منابع"""
        logger.info("🔬 collect_scientific started | query={!r}", query)

        sources = [DataSource.ARXIV, DataSource.OPENALEX, DataSource.SEMANTIC_SCHOLAR]
        results = self.search_all(query, sources=sources, max_per_source=max_results)
        return self._merge_and_deduplicate(results)

    def collect_social(
        self, 
        query: str,
        max_results: int = 20
    ) -> list[CollectedItem]:
        """جمع‌آوری اجتماعی"""
        logger.info("💬 collect_social started | query={!r}", query)

        items = []
        try:
            reddit_result = self.social.search_reddit(query, max_results=max_results)
            items.extend(reddit_result.items)
        except Exception as e:
            logger.warning(f"Reddit: {e}")

        try:
            rss_result = self.social.fetch_rss(
                self.social.POLITICAL_RSS[:3], max_per_feed=5
            )
            # فیلتر کلیدواژه‌ای — تطبیق عبارت کامل تقریباً همیشه صفر می‌شود
            items.extend(self._keyword_filter(rss_result.items, query))
        except Exception as e:
            logger.warning(f"RSS: {e}")

        return items

    def collect_financial(
        self,
        tickers: list[str],
        fred_series: list[str] | None = None,
    ) -> list[CollectedItem]:
        """جمع‌آوری مالی"""
        logger.info("💹 collect_financial started | tickers={}", tickers)

        items = []

        stock_result = self.financial.get_multiple_stocks(tickers)
        items.extend(stock_result.items)

        if fred_series and self.financial._fred:
            fred_result = self.financial.get_fred_series(fred_series)
            items.extend(fred_result.items)

        return items

    def collect_knowledge(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[CollectedItem]:
        """جمع‌آوری دانش عمومی (Wikipedia)"""
        logger.info("📚 collect_knowledge started | query={!r}", query)
        try:
            result = self.knowledge.search_wikipedia(query, max_results=max_results)
            # نتیجه لاگ شود — قبلاً «۰ آیتم» بی‌صدا رد می‌شد و معلوم نبود
            # ویکی‌پدیا چیزی برنگردانده یا در مسیر ingest گم شده
            logger.info(
                "📚 Wikipedia returned {} items for {!r} (errors={})",
                len(result.items), query, len(result.errors),
            )
            if result.errors:
                logger.warning("Wikipedia errors: {}", result.errors)
            return result.items
        except Exception as e:
            logger.warning(f"Wikipedia: {e}")
            return []

    # کلمات بی‌اثر که نباید مبنای فیلتر خبر باشند
    _FILTER_STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for",
        "i", "need", "want", "data", "about", "latest", "recent", "news",
        "information", "info", "please", "give", "me", "what", "is", "are",
        "tell", "show", "find", "get", "current", "update", "updates",
        "new",
    }

    def _keyword_filter(
        self,
        items: list[CollectedItem],
        query: str,
    ) -> list[CollectedItem]:
        """
        فیلتر آیتم‌ها بر اساس کلیدواژه‌های معنادار query.
        تطبیق عبارت کامل، queryهای محاوره‌ای را صفر می‌کند
        («I need data about iran» هیچ تیتری را match نمی‌کند).
        """
        keywords = [
            w for w in re.findall(r"[\w؀-ۿ]+", query.lower())
            if len(w) > 2 and w not in self._FILTER_STOPWORDS
        ]
        if not keywords:
            return items

        matched = [
            item for item in items
            if any(
                kw in (item.title or "").lower() or kw in (item.content or "").lower()
                for kw in keywords
            )
        ]
        logger.info(
            f"keyword filter: {len(matched)}/{len(items)} items matched keywords {keywords}"
        )
        return matched

    def collect_political_news(
        self,
        query: str | None = None,
        max_per_feed: int = 5
    ) -> list[CollectedItem]:
        """اخبار سیاسی از RSS"""
        logger.info("📰 collect_political_news started | query={!r}", query)
        result = self.social.fetch_political_news(max_per_feed=max_per_feed)
        if query:
            return self._keyword_filter(result.items, query)
        return result.items

    # ─── Pipeline یکپارچه ────────────────────────────────────────────────

    def build_rag_corpus(
        self,
        query: str,
        domains: list[DataDomain] | None = None,
        max_per_source: int = 5,
        chunk_size: int = 500,
    ) -> list[dict]:
        """
        ساخت corpus کامل برای RAG
        خروجی: لیست chunk آماده برای embed و index
        """
        domains = domains or [
            DataDomain.KNOWLEDGE,
            DataDomain.SCIENTIFIC,
            DataDomain.NEWS,
        ]

        all_items = []

        if DataDomain.KNOWLEDGE in domains:
            wiki_result = self.knowledge.search_wikipedia(query, max_results=max_per_source)
            all_items.extend(wiki_result.items)

        if DataDomain.SCIENTIFIC in domains:
            sci_items = self.collect_scientific(query, max_results=max_per_source)
            all_items.extend(sci_items)

        if DataDomain.NEWS in domains:
            try:
                rss = self.social.fetch_rss(
                    self.social.SCIENCE_RSS[:2], max_per_feed=max_per_source
                )
                all_items.extend(rss.items)
            except Exception as e:
                logger.warning(f"RSS: {e}")

        # تبدیل به chunks برای RAG
        chunks = []
        for item in all_items:
            text_chunks = DataHelper.chunk_text(item.content, chunk_size)
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "id":        f"{item.id}_chunk_{i}",
                    "text":      chunk,
                    "source":    item.source,
                    "domain":    item.domain,
                    "title":     item.title,
                    "url":       item.url,
                    "tags":      item.tags,
                    "metadata":  item.metadata,
                })

        logger.info(
            f"✅ RAG corpus: {len(all_items)} items → {len(chunks)} chunks"
        )
        return chunks

    # ─── Export ──────────────────────────────────────────────────────────

    def export_to_json(
        self,
        items: list[CollectedItem],
        filepath: str
    ) -> None:
        data = [item.model_dump(mode="json") for item in items]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 {len(items)} items → {filepath}")

    # ─── داخلی ────────────────────────────────────────────────────────────

    def _merge_and_deduplicate(
        self,
        results: dict[DataSource, CollectionResult]
    ) -> list[CollectedItem]:
        seen_ids = set()
        merged = []
        for result in results.values():
            for item in result.items:
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    merged.append(item)
        return merged

    def _print_summary_table(
        self,
        results: dict[DataSource, CollectionResult]
    ) -> None:
        table = Table(title="📊 نتایج جمع‌آوری", show_lines=True)
        table.add_column("منبع",     style="cyan",  justify="right")
        table.add_column("تعداد",    style="green", justify="center")
        table.add_column("وضعیت",    style="bold",  justify="center")
        table.add_column("زمان (s)", style="yellow",justify="center")
        table.add_column("خطا",      style="red",   justify="right")

        for source, result in results.items():
            table.add_row(
                source.value,
                str(result.total),
                "✅" if result.success else "❌",
                f"{result.elapsed_sec:.1f}",
                str(len(result.errors)) if result.errors else "-"
            )

        console.print(table)
