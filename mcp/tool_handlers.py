from __future__ import annotations

# tool_handlers.py

from external_sources.data_collector.manager  import DataCollectorManager
from external_sources.data_collector.models import DataSource, DataDomain
import logging


manager = DataCollectorManager()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────
# Search tools
# ─────────────────────────────────────

def tool_search(query: str, source: str, max_results: int = 10):
    """
    جستجو در یک منبع خاص
    """

    result = manager.search(
        query=query,
        source=DataSource(source),
        max_results=max_results
    )

    return {
        "source": source,
        "total": result.total,
        "items": [item.model_dump() for item in result.items]
    }


def tool_search_all(query: str, max_per_source: int = 5):
    """
    جستجو در همه منابع
    """

    results = manager.search_all(
        query=query,
        max_per_source=max_per_source
    )

    output = {}

    for source, result in results.items():
        output[source.value] = {
            "total": result.total,
            "items": [i.model_dump() for i in result.items]
        }

    return output


# ─────────────────────────────────────
# Scientific
# ─────────────────────────────────────

def tool_collect_scientific(query: str, max_results: int = 10):
    """
    جمع‌آوری مقالات علمی
    """

    items = manager.collect_scientific(
        query=query,
        max_results=max_results
    )

    return {
        "total": len(items),
        "items": [i.model_dump() for i in items]
    }


# ─────────────────────────────────────
# News
# ─────────────────────────────────────

def tool_collect_news(query: str | None = None, max_per_feed: int = 5):
    """
    جمع‌آوری اخبار
    """

    items = manager.collect_political_news(
        query=query,
        max_per_feed=max_per_feed
    )

    return {
        "total": len(items),
        "items": [i.model_dump() for i in items]
    }


# ─────────────────────────────────────
# Social
# ─────────────────────────────────────

def tool_collect_social(query: str, max_results: int = 20):
    """
    جمع‌آوری داده از شبکه‌های اجتماعی
    """

    items = manager.collect_social(
        query=query,
        max_results=max_results
    )
    logger.info(items)
    return {
        "total": len(items),
        "items": [i.model_dump() for i in items]
    }


# ─────────────────────────────────────
# Financial
# ─────────────────────────────────────

def tool_collect_financial(
    tickers: list[str],
    fred_series: list[str] | None = None
):
    """
    جمع‌آوری داده‌های مالی
    """

    items = manager.collect_financial(
        tickers=tickers,
        fred_series=fred_series
    )

    return {
        "total": len(items),
        "items": [i.model_dump() for i in items]
    }


# ─────────────────────────────────────
# RAG corpus
# ─────────────────────────────────────

def tool_build_rag_corpus(
    query: str,
    chunk_size: int = 500
):
    """
    ساخت corpus برای RAG
    """

    chunks = manager.build_rag_corpus(
        query=query,
        chunk_size=chunk_size
    )

    return {
        "total_chunks": len(chunks),
        "chunks": chunks
    }


# ─────────────────────────────────────
# Export
# ─────────────────────────────────────

def tool_export_json(items: list[dict], filepath: str):
    """
    export داده‌ها به JSON
    """

    from models import CollectedItem

    parsed = [CollectedItem(**item) for item in items]

    manager.export_to_json(parsed, filepath)

    return {
        "status": "saved",
        "path": filepath,
        "items": len(parsed)
    }
