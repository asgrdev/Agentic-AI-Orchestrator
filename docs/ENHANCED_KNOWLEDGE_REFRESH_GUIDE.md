# راهنمای جامع سیستم بروزرسانی دانش پیشرفته
# Enhanced Knowledge Refresh System Guide

## 📋 فهرست مطالب

1. [معرفی](#معرفی)
2. [معماری سیستم](#معماری-سیستم)
3. [مؤلفه‌های اصلی](#مؤلفه‌های-اصلی)
4. [فرآیند بروزرسانی](#فرآیند-بروزرسانی)
5. [گراف معنایی](#گراف-معنایی)
6. [استفاده](#استفاده)
7. [پیکربندی](#پیکربندی)
8. [متریک‌ها و نظارت](#متریک‌ها-و-نظارت)
9. [بهینه‌سازی](#بهینه‌سازی)
10. [عیب‌یابی](#عیب‌یابی)

---

## معرفی

سیستم بروزرسانی دانش پیشرفته (Enhanced Knowledge Refresh System) یک راه‌حل یکپارچه برای جمع‌آوری، پردازش و ادغام دانش جدید در گراف دانش است.

### ویژگی‌های کلیدی

✅ **یکپارچگی با گراف معنایی**: استفاده از SemanticGraphBuilder برای ساخت روابط معنایی غنی

✅ **بروزرسانی تدریجی**: ادغام هوشمند دانش جدید با دانش موجود

✅ **حل تعارض خودکار**: تشخیص و حل تعارض‌های دانشی

✅ **نسخه‌بندی**: نگهداری تاریخچه تغییرات

✅ **استنتاج روابط**: تولید خودکار روابط جدید از روابط موجود

✅ **اعتبارسنجی کیفیت**: ارزیابی مستمر کیفیت گراف

✅ **متریک‌های جامع**: نظارت دقیق بر فرآیند بروزرسانی

---

## معماری سیستم

```
┌─────────────────────────────────────────────────────────────┐
│                  Enhanced Knowledge Refresh                  │
│                         Agent                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         Data Collection Layer            │
        ├─────────────────────────────────────────┤
        │  • Financial Data (Tickers)              │
        │  • Knowledge Sources (Wikipedia, etc.)   │
        │  • Scientific Papers                     │
        │  • News & Social Media                   │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │       Processing & Extraction Layer      │
        ├─────────────────────────────────────────┤
        │  • Chunk Creation                        │
        │  • Entity Extraction (NER)               │
        │  • Relation Extraction (LLM)             │
        │  • Embedding Generation                  │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Semantic Graph Builder Layer        │
        ├─────────────────────────────────────────┤
        │  • Entity Canonicalization               │
        │  • Relation Normalization                │
        │  • Conflict Resolution                   │
        │  • Relation Inference                    │
        │  • Versioning                            │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │          Storage Layer                   │
        ├─────────────────────────────────────────┤
        │  • KuzuDB (Knowledge Graph)              │
        │  • Weaviate (Vector Store)               │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │       Quality Validation Layer           │
        ├─────────────────────────────────────────┤
        │  • Graph Density Analysis                │
        │  • Connectivity Metrics                  │
        │  • Quality Scoring                       │
        └─────────────────────────────────────────┘
```

---

## مؤلفه‌های اصلی

### 1. EnhancedKnowledgeRefreshAgent

Agent اصلی که تمام فرآیند بروزرسانی را مدیریت می‌کند.

```python
from agents.enhanced_knowledge_refresh_agent import EnhancedKnowledgeRefreshAgent

config = {
    "kuzu_path": "data/kuzu_db",
    "embed_client": embed_client,
    "refresh_interval": 60,
    "enable_inference": True,
    "enable_versioning": True,
}

agent = EnhancedKnowledgeRefreshAgent(config)
await agent.startup()
```

### 2. SemanticGraphBuilder

سازنده گراف معنایی با قابلیت‌های پیشرفته:

```python
from ingestion.semantic_graph_builder import SemanticGraphBuilder

builder = SemanticGraphBuilder(
    graph_manager=kuzu_manager,
    enable_inference=True,
    enable_versioning=True,
    confidence_threshold=0.5,
)

# ساخت گراف معنایی
update = await builder.build_semantic_graph(
    chunk=chunk_data,
    entities=entities,
    relations=relations,
)
```

### 3. Data Collectors

جمع‌آوری داده از منابع مختلف:

- **Financial**: داده‌های مالی و بورسی
- **Knowledge**: Wikipedia, DBpedia
- **Scientific**: arXiv, PubMed
- **News**: اخبار سیاسی و اقتصادی
- **Social**: شبکه‌های اجتماعی

### 4. Processing Pipelines

- **EntityExtractionPipeline**: استخراج entities با NER
- **RelationPipeline**: استخراج relations با LLM
- **ChunkIngestionPipeline**: پردازش و ذخیره chunks

---

## فرآیند بروزرسانی

### مراحل اصلی

#### 1️⃣ جمع‌آوری داده (Data Collection)

```python
# استخراج tickers از query
tickers = agent._extract_tickers(query)

# جمع‌آوری از منابع مختلف
items = await agent._collect_data(query, metrics)
```

**منابع داده:**
- Financial: اگر ticker پیدا شود
- Knowledge: همیشه
- News: همیشه
- Scientific: اختیاری
- Social: اختیاری

#### 2️⃣ پردازش و استخراج (Processing & Extraction)

```python
# پردازش هر item
for item in items:
    # ایجاد chunk
    chunk = create_chunk(item)
    
    # استخراج entities
    entities = await entity_pipeline.extract(text)
    
    # استخراج relations
    relations = await relation_pipeline.extract(text, entities)
```

**خروجی:**
```python
{
    "chunk": {
        "id": "chunk_123",
        "title": "...",
        "text": "...",
    },
    "entities": [Entity(...)],
    "relations": [Relation(...)],
}
```

#### 3️⃣ بروزرسانی گراف معنایی (Semantic Graph Update)

```python
# ساخت گراف معنایی
update = await semantic_builder.build_semantic_graph(
    chunk=chunk,
    entities=entities,
    relations=relations,
)
```

**عملیات انجام شده:**

1. **پردازش Entities:**
   - جستجوی entity موجود
   - ادغام یا ایجاد جدید
   - به‌روزرسانی confidence
   - نسخه‌بندی

2. **پردازش Relations:**
   - جستجوی relation موجود
   - حل تعارض
   - به‌روزرسانی confidence
   - نگهداری شواهد

3. **استنتاج روابط:**
   - Transitive relations (A→B, B→C ⇒ A→C)
   - Symmetric relations (A→B ⇒ B→A)
   - Hierarchical relations (instance_of)

4. **درج در گراف:**
   - Batch upsert entities
   - Batch upsert relations
   - ایجاد MENTIONS edges

#### 4️⃣ اعتبارسنجی کیفیت (Quality Validation)

```python
quality_score = await agent._validate_graph_quality()
```

**معیارهای کیفیت:**
- نسبت روابط به entities
- تراکم گراف
- اتصال‌پذیری

**محاسبه امتیاز:**
```python
relation_ratio = relation_count / entity_count

if relation_ratio < 1:
    quality = relation_ratio * 0.5  # Low connectivity
elif relation_ratio <= 3:
    quality = 0.5 + (relation_ratio - 1) * 0.25  # Good
elif relation_ratio <= 5:
    quality = 1.0  # Optimal
else:
    quality = max(0.8, 1.0 - (relation_ratio - 5) * 0.05)  # Too dense
```

#### 5️⃣ ذخیره متریک‌ها (Metrics Storage)

```python
metrics = RefreshMetrics(
    total_items=100,
    chunks_ingested=95,
    nodes_added=50,
    nodes_updated=30,
    edges_added=120,
    edges_updated=40,
    inferred_relations=25,
    conflicts_resolved=10,
    quality_score=0.85,
    duration_seconds=12.5,
)
```

---

## گراف معنایی

### انواع روابط

```python
class RelationType(Enum):
    EXPLICIT = "explicit"          # روابط صریح از متن
    INFERRED = "inferred"          # روابط استنتاجی
    SEMANTIC = "semantic"          # روابط معنایی
    TEMPORAL = "temporal"          # روابط زمانی
    CAUSAL = "causal"             # روابط علت و معلولی
    HIERARCHICAL = "hierarchical"  # روابط سلسله‌مراتبی
```

### استنتاج روابط

#### 1. Transitive Relations

```
A --part_of--> B
B --part_of--> C
─────────────────
A --part_of--> C  (inferred)
```

**مثال:**
```
Tehran --located_in--> Iran
Iran --located_in--> Asia
────────────────────────────
Tehran --located_in--> Asia  (inferred)
```

#### 2. Symmetric Relations

```
A --similar_to--> B
───────────────────
B --similar_to--> A  (inferred)
```

**مثال:**
```
Python --similar_to--> Ruby
───────────────────────────
Ruby --similar_to--> Python  (inferred)
```

#### 3. Hierarchical Relations

```
Entity: "Python"
Type: "ProgrammingLanguage"
────────────────────────────
Python --instance_of--> TYPE:ProgrammingLanguage  (inferred)
```

### حل تعارض

```python
# تعارض: دو relation با confidence متفاوت
existing: confidence = 0.7
new:      confidence = 0.9

# استراتژی: میانگین وزن‌دار
if abs(new_conf - old_conf) > 0.3:
    merged = old_conf * 0.4 + new_conf * 0.6  # وزن بیشتر به جدید
else:
    merged = old_conf * 0.6 + new_conf * 0.4  # وزن بیشتر به قدیم
```

### نسخه‌بندی

```python
# نسخه اول
entity = {
    "id": "entity_123",
    "name": "Python",
    "version": 1,
    "previous_names": [],
}

# بروزرسانی
entity = {
    "id": "entity_123",
    "name": "Python Programming Language",
    "version": 2,
    "previous_names": ["Python"],
}
```

---

## استفاده

### راه‌اندازی اولیه

```python
from agents.enhanced_knowledge_refresh_agent import EnhancedKnowledgeRefreshAgent
from ingestion.embedding_generator import Qwen3EmbeddingClient
from external_sources.data_collector.manager import DataCollectorManager
from external_sources.data_collector.models import DataDomain

# 1. ایجاد embedding client
embed_client = Qwen3EmbeddingClient(
    model_path="models/Qwen3-Embedding-0.6B"
)

# 2. ایجاد data collector
data_manager = DataCollectorManager()

# 3. پیکربندی agent
config = {
    "kuzu_path": "data/kuzu_db",
    "embed_client": embed_client,
    "data_manager": data_manager,
    "collection_name": "Chunks",
    "refresh_interval": 60,
    "max_concurrent": 5,
    "enable_inference": True,
    "enable_versioning": True,
    "confidence_threshold": 0.5,
    "quality_threshold": 0.6,
    "domains": [
        DataDomain.NEWS,
        DataDomain.KNOWLEDGE,
        DataDomain.SCIENTIFIC,
    ],
    "weaviate": {
        "mode": "docker",
        "host": "localhost",
        "port": 8080,
    },
}

# 4. ایجاد agent
agent = EnhancedKnowledgeRefreshAgent(config)

# 5. راه‌اندازی
await agent.startup()
```

### بروزرسانی دانش

```python
from agents.state import AgentState

# ایجاد state
state = AgentState(
    query="What is the latest news about Tesla stock?",
    refresh_needed=True,
)

# بروزرسانی
updated_state = await agent.refresh(state)

# بررسی نتایج
if updated_state.refresh_needed:
    print("Refresh was throttled or failed")
else:
    metrics = updated_state.metadata.get("refresh_metrics", {})
    print(f"Chunks ingested: {metrics['chunks_ingested']}")
    print(f"Nodes added: {metrics['nodes_added']}")
    print(f"Edges added: {metrics['edges_added']}")
    print(f"Quality score: {metrics['quality_score']:.2f}")
```

### دریافت تاریخچه

```python
# تاریخچه 10 بروزرسانی اخیر
history = agent.get_refresh_history(limit=10)

for entry in history:
    print(f"Time: {entry['timestamp']}")
    print(f"Items: {entry['total_items']}")
    print(f"Quality: {entry['quality_score']:.2f}")
    print(f"Duration: {entry['duration']:.1f}s")
    print("---")
```

### آمار گراف

```python
stats = await agent.get_graph_stats()
print(f"Entities: {stats['entity_count']}")
print(f"Relations: {stats['relation_count']}")
```

---

## پیکربندی

### پارامترهای اصلی

| پارامتر | نوع | پیش‌فرض | توضیحات |
|---------|-----|---------|---------|
| `kuzu_path` | str | - | مسیر دیتابیس KuzuDB |
| `embed_client` | object | - | Client برای embedding |
| `refresh_interval` | int | 60 | فاصله بین بروزرسانی‌ها (ثانیه) |
| `max_concurrent` | int | 5 | تعداد task های همزمان |
| `enable_inference` | bool | True | فعال‌سازی استنتاج روابط |
| `enable_versioning` | bool | True | فعال‌سازی نسخه‌بندی |
| `confidence_threshold` | float | 0.5 | حداقل confidence برای روابط |
| `quality_threshold` | float | 0.6 | حداقل کیفیت مورد انتظار |
| `domains` | list | [NEWS, KNOWLEDGE] | منابع داده فعال |

### پیکربندی Weaviate

```python
"weaviate": {
    "mode": "docker",        # docker, cloud, embedded
    "host": "localhost",
    "port": 8080,
    "grpc_port": 50051,
    "cloud_url": "",         # برای cloud mode
    "api_key": "",           # برای cloud mode
}
```

### پیکربندی Domains

```python
from external_sources.data_collector.models import DataDomain

"domains": [
    DataDomain.KNOWLEDGE,    # Wikipedia, DBpedia
    DataDomain.SCIENTIFIC,   # arXiv, PubMed
    DataDomain.NEWS,         # News sources
    DataDomain.SOCIAL,       # Social media
]
```

---

## متریک‌ها و نظارت

### RefreshMetrics

```python
@dataclass
class RefreshMetrics:
    total_items: int              # تعداد کل items جمع‌آوری شده
    chunks_ingested: int          # تعداد chunks پردازش شده
    nodes_added: int              # تعداد entities جدید
    nodes_updated: int            # تعداد entities بروزرسانی شده
    edges_added: int              # تعداد relations جدید
    edges_updated: int            # تعداد relations بروزرسانی شده
    inferred_relations: int       # تعداد روابط استنتاجی
    conflicts_resolved: int       # تعداد تعارض‌های حل شده
    quality_score: float          # امتیاز کیفیت (0-1)
    duration_seconds: float       # مدت زمان بروزرسانی
    errors: list[str]             # لیست خطاها
```

### نظارت در Production

```python
import logging

# فعال‌سازی logging
logging.basicConfig(level=logging.INFO)

# نظارت بر metrics
async def monitor_refresh():
    while True:
        history = agent.get_refresh_history(limit=1)
        if history:
            latest = history[0]
            
            # هشدار برای کیفیت پایین
            if latest['quality_score'] < 0.6:
                logger.warning(f"Low quality: {latest['quality_score']}")
            
            # هشدار برای خطاها
            if latest['errors'] > 0:
                logger.error(f"Errors detected: {latest['errors']}")
        
        await asyncio.sleep(300)  # هر 5 دقیقه
```

---

## بهینه‌سازی

### 1. تنظیم Concurrency

```python
# برای سیستم‌های قوی
config["max_concurrent"] = 10

# برای سیستم‌های ضعیف
config["max_concurrent"] = 2
```

### 2. تنظیم Confidence Threshold

```python
# برای دقت بالاتر (روابط کمتر اما دقیق‌تر)
config["confidence_threshold"] = 0.7

# برای پوشش بیشتر (روابط بیشتر اما کم‌دقت‌تر)
config["confidence_threshold"] = 0.3
```

### 3. غیرفعال کردن Inference

```python
# برای سرعت بیشتر
config["enable_inference"] = False
```

### 4. پاک کردن Cache

```python
# پاک کردن cache برای آزادسازی حافظه
agent.clear_cache()
```

### 5. Batch Size

```python
# در ChunkIngestionPipeline
config["chunk_size"] = 400  # کوچک‌تر برای دقت بیشتر
config["chunk_size"] = 800  # بزرگ‌تر برای سرعت بیشتر
```

---

## عیب‌یابی

### مشکل: بروزرسانی انجام نمی‌شود

**علت:** Throttling فعال است

**راه‌حل:**
```python
# کاهش refresh_interval
config["refresh_interval"] = 30

# یا پاک کردن cache
agent._last_refresh.clear()
```

### مشکل: کیفیت گراف پایین است

**علت:** روابط کم یا زیاد

**راه‌حل:**
```python
# بررسی آمار
stats = await agent.get_graph_stats()
ratio = stats['relation_count'] / stats['entity_count']

if ratio < 1:
    # روابط کم - کاهش confidence_threshold
    config["confidence_threshold"] = 0.3
elif ratio > 5:
    # روابط زیاد - افزایش confidence_threshold
    config["confidence_threshold"] = 0.7
```

### مشکل: خطاهای مکرر در collection

**علت:** منابع داده در دسترس نیستند

**راه‌حل:**
```python
# غیرفعال کردن domains مشکل‌دار
config["domains"] = [DataDomain.KNOWLEDGE]  # فقط Wikipedia

# یا بررسی logs
history = agent.get_refresh_history(limit=5)
for entry in history:
    if entry['errors'] > 0:
        print(f"Errors at {entry['timestamp']}")
```

### مشکل: حافظه پر می‌شود

**علت:** Cache بزرگ

**راه‌حل:**
```python
# پاک کردن cache به صورت دوره‌ای
async def periodic_cache_clear():
    while True:
        await asyncio.sleep(3600)  # هر ساعت
        agent.clear_cache()
        logger.info("Cache cleared")
```

### مشکل: بروزرسانی خیلی کند است

**علت:** Inference یا Versioning فعال است

**راه‌حل:**
```python
# غیرفعال کردن inference
config["enable_inference"] = False

# غیرفعال کردن versioning
config["enable_versioning"] = False

# افزایش concurrency
config["max_concurrent"] = 10
```

---

## مثال‌های کاربردی

### مثال 1: بروزرسانی ساده

```python
# Setup
agent = EnhancedKnowledgeRefreshAgent(config)
await agent.startup()

# Refresh
state = AgentState(query="Tesla stock news", refresh_needed=True)
state = await agent.refresh(state)

# Results
print(f"Quality: {state.metadata['refresh_metrics']['quality_score']}")
```

### مثال 2: نظارت مستمر

```python
async def continuous_monitoring():
    agent = EnhancedKnowledgeRefreshAgent(config)
    await agent.startup()
    
    queries = [
        "Tesla stock",
        "Apple earnings",
        "Bitcoin price",
    ]
    
    while True:
        for query in queries:
            state = AgentState(query=query, refresh_needed=True)
            state = await agent.refresh(state)
            
            metrics = state.metadata.get("refresh_metrics", {})
            logger.info(f"Query: {query}, Quality: {metrics.get('quality_score', 0)}")
        
        await asyncio.sleep(300)  # هر 5 دقیقه
```

### مثال 3: بروزرسانی با فیلتر کیفیت

```python
async def quality_filtered_refresh(query: str, min_quality: float = 0.7):
    agent = EnhancedKnowledgeRefreshAgent(config)
    await agent.startup()
    
    max_retries = 3
    for attempt in range(max_retries):
        state = AgentState(query=query, refresh_needed=True)
        state = await agent.refresh(state)
        
        quality = state.metadata.get("refresh_metrics", {}).get("quality_score", 0)
        
        if quality >= min_quality:
            logger.info(f"Quality threshold met: {quality:.2f}")
            return state
        
        logger.warning(f"Attempt {attempt + 1}: Quality {quality:.2f} < {min_quality}")
        await asyncio.sleep(10)
    
    logger.error("Failed to meet quality threshold")
    return state
```

---

## نتیجه‌گیری

سیستم بروزرسانی دانش پیشرفته یک راه‌حل جامع برای:

✅ جمع‌آوری خودکار دانش از منابع متنوع
✅ پردازش هوشمند و استخراج entities/relations
✅ ساخت گراف معنایی با روابط غنی
✅ بروزرسانی تدریجی و حل تعارض
✅ استنتاج روابط جدید
✅ اعتبارسنجی کیفیت
✅ نظارت و متریک‌های جامع

### مزایا

- **مقیاس‌پذیری**: پشتیبانی از concurrent processing
- **انعطاف‌پذیری**: پیکربندی آسان برای نیازهای مختلف
- **قابلیت اطمینان**: error handling و retry mechanism
- **شفافیت**: متریک‌های جامع و logging دقیق
- **کیفیت**: اعتبارسنجی مستمر و بهینه‌سازی خودکار

### استفاده در Production

برای استفاده در محیط production:

1. تنظیم logging مناسب
2. نظارت بر metrics
3. تنظیم thresholds بر اساس نیاز
4. پاک‌سازی دوره‌ای cache
5. backup منظم از گراف دانش

---
 