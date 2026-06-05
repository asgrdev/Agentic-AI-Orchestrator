from manager import DataCollectorManager
from models import DataSource, DataDomain

# ─── راه‌اندازی ─────────────────────────────────────────────────────────
manager = DataCollectorManager(
    reddit_client_id="YOUR_ID",
    reddit_client_secret="YOUR_SECRET",
    fred_api_key="YOUR_FRED_KEY",   # اختیاری
)

# ─── ۱. جستجوی علمی ─────────────────────────────────────────────────────
sci_items = manager.collect_scientific(
    query="graph neural networks knowledge graph",
    max_results=5
)
print(f"\n📚 {len(sci_items)} مقاله یافت شد")
for item in sci_items[:3]:
    print(f"  • {item.title} [{item.source}]")

# ─── ۲. اخبار سیاسی ─────────────────────────────────────────────────────
political = manager.collect_political_news(
    query="AI regulation",
    max_per_feed=3
)
print(f"\n🏛️  {len(political)} خبر سیاسی")

# ─── ۳. داده‌های مالی ────────────────────────────────────────────────────
financial = manager.collect_financial(
    tickers=["AAPL", "GOOGL", "MSFT"],
    fred_series=["GDP", "CPIAUCSL", "UNRATE"]   # نیاز به API key
)
print(f"\n💰 {len(financial)} آیتم مالی")

# ─── ۴. جستجو در همه منابع به صورت موازی ───────────────────────────────
results = manager.search_all(
    query="large language models",
    sources=[
        DataSource.WIKIPEDIA,
        DataSource.ARXIV,
        DataSource.OPENALEX,
        DataSource.REDDIT,
    ],
    max_per_source=5
)

# ─── ۵. ساخت corpus برای RAG ─────────────────────────────────────────────
chunks = manager.build_rag_corpus(
    query="artificial intelligence safety",
    domains=[DataDomain.KNOWLEDGE, DataDomain.SCIENTIFIC],
    max_per_source=3,
    chunk_size=400,
)
print(f"\n🧩 {len(chunks)} chunk آماده برای RAG")

# ─── ۶. export ───────────────────────────────────────────────────────────
all_items = []
for result in results.values():
    all_items.extend(result.items)

manager.export_to_json(all_items, "collected_data.json")
