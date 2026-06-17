# بهبود سیستم بروزرسانی دانش (Knowledge Refresh Improvement)

## خلاصه تغییرات

این سند تغییرات و بهبودهای اعمال شده بر سیستم بروزرسانی دانش و گراف معنایی را شرح می‌دهد.

## 🎯 اهداف

1. **بهبود فرآیند بروزرسانی دانش**: از بازنویسی کامل به بروزرسانی تدریجی
2. **گراف معنایی پیشرفته**: ایجاد روابط معنایی غنی‌تر و استنتاج روابط جدید
3. **حل تعارض هوشمند**: مدیریت تعارضات بین دانش جدید و موجود
4. **نسخه‌بندی دانش**: نگهداری تاریخچه تغییرات
5. **متریک‌های کیفیت**: ارزیابی کیفیت گراف دانش

## 📁 فایل‌های جدید

### 1. `ingestion/semantic_graph_builder.py`

**هدف**: ساخت گراف معنایی با قابلیت‌های پیشرفته

**ویژگی‌های کلیدی**:

#### انواع روابط (RelationType)
```python
class RelationType(Enum):
    EXPLICIT = "explicit"          # روابط صریح از متن
    INFERRED = "inferred"          # روابط استنتاجی
    SEMANTIC = "semantic"          # روابط معنایی (مترادف، ضد، ...)
    TEMPORAL = "temporal"          # روابط زمانی
    CAUSAL = "causal"             # روابط علت و معلولی
    HIERARCHICAL = "hierarchical"  # روابط سلسله‌مراتبی
```

#### کلاس SemanticRelation
روابط با متادیتای کامل شامل:
- `subject_id`, `predicate`, `object_id`: اجزای اصلی رابطه
- `relation_type`: نوع رابطه
- `confidence`: اطمینان از رابطه
- `version`: نسخه رابطه (برای tracking تغییرات)
- `evidence`: شواهد متنی
- `timestamp`: زمان ایجاد/بروزرسانی

#### متدهای اصلی

##### `build_semantic_graph()`
ساخت گراف معنایی کامل از chunk + entities + relations

```python
async def build_semantic_graph(
    self,
    chunk: dict,
    entities: list[Entity],
    relations: list[Relation],
) -> GraphUpdate
```

**فرآیند**:
1. پردازش و ادغام entities
2. پردازش و ادغام relations
3. استنتاج روابط جدید (اگر فعال باشد)
4. درج در گراف
5. ایجاد روابط معنایی

##### `incremental_update()`
بروزرسانی تدریجی با حل تعارض

```python
async def incremental_update(
    self,
    new_entities: list[Entity],
    new_relations: list[Relation],
    chunk: dict,
) -> GraphUpdate
```

**مزایا**:
- فقط تغییرات را اعمال می‌کند (کارآمدتر)
- حل تعارض هوشمند
- نگهداری تاریخچه

#### استنتاج روابط (Relation Inference)

##### روابط متعدی (Transitive)
```
A → B و B → C ⇒ A → C
```
مثال: اگر "تهران" در "ایران" و "ایران" در "آسیا" باشد، پس "تهران" در "آسیا" است.

##### روابط متقارن (Symmetric)
```
A → B ⇒ B → A
```
مثال: اگر "علی" همکار "رضا" است، پس "رضا" همکار "علی" است.

##### روابط سلسله‌مراتبی (Hierarchical)
ایجاد روابط `instance_of` بین entities و type های آنها.

#### حل تعارض (Conflict Resolution)

استراتژی: **میانگین وزن‌دار confidence**

```python
# اگر تفاوت زیاد باشد، وزن بیشتری به جدید
if abs(new_conf - old_conf) > 0.3:
    merged_conf = (old_conf * 0.4 + new_conf * 0.6)
else:
    merged_conf = (old_conf * 0.6 + new_conf * 0.4)
```

### 2. `agents/improved_knowledge_refresh_agent.py`

**هدف**: Agent بهبود یافته برای بروزرسانی دانش

**تغییرات نسبت به نسخه قبل**:

#### 1. استفاده از SemanticGraphBuilder
```python
self._semantic_builder = SemanticGraphBuilder(
    graph_manager=self._kuzu,
    enable_inference=True,
    enable_versioning=True,
    confidence_threshold=0.5,
)
```

#### 2. مدیریت حافظه بهتر
استفاده از `ModelWrapper` برای مدیریت مدل‌های embedding:

```python
if self._use_model_manager:
    self._embed_wrapper = get_embedding_model()
    # مدل فقط هنگام نیاز load می‌شود
```

#### 3. متریک‌های جامع

کلاس `RefreshMetrics` برای tracking:
- تعداد items و chunks
- nodes و edges اضافه/بروزرسانی شده
- روابط استنتاجی
- تعارضات حل شده
- مدت زمان

#### 4. بروزرسانی تدریجی

متد `incremental_refresh()` برای بروزرسانی‌های کوچک:

```python
async def incremental_refresh(
    self,
    state: AgentState,
    new_data: list,
) -> GraphUpdate
```

**مزایا**:
- سریعتر از refresh کامل
- کمتر resource-intensive
- حفظ consistency گراف

#### 5. متریک‌های کیفیت گراف

```python
async def get_graph_quality_metrics(self) -> dict:
    """
    Returns:
        - entity_count: تعداد entities
        - relation_count: تعداد relations
        - graph_density: تراکم گراف
        - average_degree: میانگین درجه nodes
        - refresh_count: تعداد بروزرسانی‌ها
    """
```

## 🔄 فلوی بهبود یافته

### فلوی قبلی (Old Flow)
```
Query → Data Collection → Chunking → Embedding → 
Vector Store + Simple Graph → Retrieval
```

**مشکلات**:
- بازنویسی کامل در هر refresh
- روابط ساده بدون معنا
- عدم حل تعارض
- بدون استنتاج

### فلوی جدید (New Flow)
```
Query → Data Collection → Chunking → 
Entity Extraction → Relation Extraction →
Semantic Graph Building (با inference) →
Incremental Update (با conflict resolution) →
Vector Store + Rich Semantic Graph → 
Enhanced Retrieval
```

**بهبودها**:
- ✅ بروزرسانی تدریجی
- ✅ روابط معنایی غنی
- ✅ استنتاج روابط جدید
- ✅ حل تعارض هوشمند
- ✅ نسخه‌بندی
- ✅ متریک‌های کیفیت

## 📊 مثال استفاده

### 1. راه‌اندازی Agent

```python
from agents.improved_knowledge_refresh_agent import ImprovedKnowledgeRefreshAgent

config = {
    "kuzu_path": "data/kuzu_db",
    "weaviate": {
        "mode": "docker",
        "host": "localhost",
        "port": 8080,
    },
    "use_model_manager": True,
    "enable_inference": True,
    "enable_versioning": True,
    "confidence_threshold": 0.5,
    "refresh_interval": 60,
    "max_concurrent": 5,
    "domains": [DataDomain.NEWS, DataDomain.KNOWLEDGE],
}

agent = ImprovedKnowledgeRefreshAgent(config)
await agent.startup()
```

### 2. بروزرسانی کامل

```python
from agents.state import AgentState

state = AgentState(query="آخرین اخبار بورس تهران")
state.refresh_needed = True

# بروزرسانی
state = await agent.refresh(state)

# بررسی متریک‌ها
metrics = state.metadata.get("last_refresh_metrics")
print(f"Chunks ingested: {metrics['chunks_ingested']}")
print(f"Nodes added: {metrics['nodes_added']}")
print(f"Edges added: {metrics['edges_added']}")
```

### 3. بروزرسانی تدریجی

```python
# دریافت داده جدید
new_items = data_collector.collect_news(query="بورس", limit=10)

# بروزرسانی تدریجی
update = await agent.incremental_refresh(state, new_items)

print(f"Nodes updated: {update.nodes_updated}")
print(f"Conflicts resolved: {update.conflicts_resolved}")
print(f"Inferred relations: {update.inferred_relations}")
```

### 4. متریک‌های کیفیت

```python
# دریافت متریک‌های گراف
quality = await agent.get_graph_quality_metrics()

print(f"Entity count: {quality['entity_count']}")
print(f"Relation count: {quality['relation_count']}")
print(f"Graph density: {quality['graph_density']}")
print(f"Average degree: {quality['average_degree']}")
```

### 5. تاریخچه بروزرسانی‌ها

```python
# دریافت 10 بروزرسانی اخیر
history = agent.get_refresh_history(limit=10)

for refresh in history:
    print(f"Time: {refresh['timestamp']}")
    print(f"Duration: {refresh['duration_seconds']}s")
    print(f"Items: {refresh['total_items']}")
    print("---")
```

## 🔍 استفاده در Retrieval

گراف معنایی بهبود یافته در `RetrieverAgent` استفاده می‌شود:

```python
# در RetrieverAgent
async def retrieve(self, state: AgentState) -> None:
    # ... vector search ...
    
    # دریافت subgraph با روابط معنایی
    graph_nodes, graph_edges = await self._fetch_subgraph(entity_ids)
    
    # graph_edges حالا شامل:
    # - روابط صریح از متن
    # - روابط استنتاجی
    # - روابط معنایی
    # - روابط سلسله‌مراتبی
    
    state.retrieval = RetrievalContext(
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        # ...
    )
```

## 📈 مزایای سیستم جدید

### 1. کارایی بهتر
- بروزرسانی تدریجی به جای کامل
- کش برای جلوگیری از query های تکراری
- مدیریت بهتر حافظه با Model Manager

### 2. کیفیت بالاتر
- روابط معنایی غنی‌تر
- استنتاج روابط جدید
- حل تعارض هوشمند
- نسخه‌بندی برای tracking

### 3. قابلیت نگهداری
- کد modular و قابل توسعه
- متریک‌های جامع
- logging کامل
- documentation

### 4. مقیاس‌پذیری
- موازی‌سازی با semaphore
- batch operations
- incremental updates

## 🔧 پیکربندی پیشرفته

### تنظیم Inference

```python
config = {
    "enable_inference": True,
    "inference_rules": {
        "transitive": ["part_of", "located_in", "member_of"],
        "symmetric": ["similar_to", "related_to", "colleague_of"],
        "hierarchical": True,
    }
}
```

### تنظیم Conflict Resolution

```python
config = {
    "conflict_strategy": "weighted_average",  # یا "newest", "highest_confidence"
    "confidence_threshold": 0.5,
    "version_retention": 5,  # نگهداری 5 نسخه قبلی
}
```

### تنظیم Performance

```python
config = {
    "max_concurrent": 5,  # تعداد task های موازی
    "refresh_interval": 60,  # ثانیه
    "cache_size": 1000,  # تعداد entities در کش
    "batch_size": 100,  # تعداد nodes در هر batch
}
```

## 🐛 عیب‌یابی

### مشکل: Refresh خیلی کند است

**راه‌حل**:
1. افزایش `max_concurrent`
2. استفاده از `incremental_refresh` به جای `refresh`
3. کاهش `graph_hops` در retrieval
4. فعال کردن caching

### مشکل: حافظه زیاد مصرف می‌شود

**راه‌حل**:
1. فعال کردن `use_model_manager=True`
2. کاهش `batch_size`
3. پاک کردن کش: `agent._semantic_builder.clear_cache()`
4. استفاده از `MemoryMonitor`

### مشکل: تعارضات زیاد

**راه‌حل**:
1. افزایش `confidence_threshold`
2. بررسی کیفیت entity extraction
3. تنظیم `conflict_strategy`
4. بررسی logs برای pattern های تعارض

## 📚 منابع بیشتر

- [Graph RAG Paper](https://arxiv.org/abs/2404.16130)
- [Knowledge Graph Embedding](https://arxiv.org/abs/1503.00759)
- [Semantic Web Technologies](https://www.w3.org/standards/semanticweb/)

## 🔮 توسعه‌های آینده

1. **Graph Neural Networks**: استفاده از GNN برای embedding گراف
2. **Temporal Reasoning**: استدلال زمانی روی گراف
3. **Multi-hop Reasoning**: استدلال چند مرحله‌ای
4. **Graph Visualization**: نمایش بصری گراف
5. **Federated Learning**: یادگیری فدرال روی گراف‌های توزیع شده

## 📝 نتیجه‌گیری

سیستم بهبود یافته بروزرسانی دانش:
- ✅ کارآمدتر و سریعتر
- ✅ گراف معنایی غنی‌تر
- ✅ قابل نگهداری و توسعه
- ✅ مقیاس‌پذیر

این بهبودها کیفیت پاسخ‌های سیستم RAG را به طور قابل توجهی افزایش می‌دهند.