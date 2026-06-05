import asyncio
from ingestion.embedding_generator import Qwen3EmbeddingClient


def main():
    # ─── راه‌اندازی ───
    client = Qwen3EmbeddingClient(
        #model_path=None,        # None = دانلود خودکار از HuggingFace
        model_path="/Users/dbk/Desktop/agentic-graph-RAG/models/Qwen3-Embedding-0.6B",  # یا مسیر لوکال
        device=None,            # None = تشخیص خودکار
        use_fp16=True,          # صرفه‌جویی در VRAM
        use_4bit=False,         # کوانتیزیشن 4-bit (نیاز به bitsandbytes)
        batch_size=16,
        pooling_mode="last_token",
        cache_embeddings=True,
        normalize=True,
    )

    print(client)
    print("اطلاعات مدل:", client.model_info())

    # ─── تک متن ───
    result = client.embed_single("هوش مصنوعی آینده بشریت را تغییر می‌دهد")
    print(f"\n✅ embedding ساخته شد")
    print(f"   ابعاد : {len(result['embedding'])}")
    print(f"   توکن‌ها: {result['tokens']}")
    print(f"   زمان  : {result['time_ms']} ms")

    # ─── چند متن ───
    texts = [
        "یادگیری ماشین شاخه‌ای از هوش مصنوعی است",
        "شبکه‌های عصبی از نورون‌های مصنوعی ساخته می‌شوند",
        "پایتون محبوب‌ترین زبان برای data science است",
        "امروز هوا خوب است",
        "آمار فعالیت این ماه شرکت به شرح زیر است : هیچ فعالیتی کشف نشد!"
        "سیب میوه‌ای سالم است",
    ]

    batch = client.embed_batch(texts)
    print(f"\n📦 Batch پردازش شد")
    print(f"   تعداد : {len(batch['results'])}")
    print(f"   توکن کل: {batch['total_tokens']}")
    print(f"   زمان کل: {batch['total_time_ms']} ms")

    # ─── مقایسه و رتبه‌بندی ───
    query  = client.embed_single("i need analysis data for activation statistic")
    # print(f"dim={(query['embedding'])}  | {query['time_ms']}ms")

    corpus = [r["embedding"] for r in batch["results"]]

    ranked = client.rank_by_similarity(
        query_embedding=query["embedding"],
        candidate_embeddings=corpus,
        texts=texts,
        top_k=3,
    )

    print("\n🔍 نزدیک‌ترین متن‌ها به query:")
    for item in ranked:
        print(f"   [{item['rank']}] score={item['score']:.4f} | {item['text']}")

    # ─── ماتریس شباهت ───
    matrix = client.similarity_matrix(corpus[:3])
    print("\n📊 ماتریس شباهت (3×3):")
    print(matrix.round(3))

    # ─── اجرای async ───
    async def async_demo():
        r = await client.async_embed_single("تست async")
        print(f"\n⚡ async OK | tokens={r['tokens']}")

    asyncio.run(async_demo())

    # ─── اطلاعات کش ───
    print(f"\n💾 کش: {client.cache_size()} آیتم")
    client.clear_cache()


if __name__ == "__main__":
    main()
