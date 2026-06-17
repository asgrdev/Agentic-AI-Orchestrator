# راهنمای نصب و پیکربندی KuzuDB

## 📋 فهرست مطالب
1. [معرفی](#معرفی)
2. [نصب](#نصب)
3. [پیکربندی](#پیکربندی)
4. [استفاده](#استفاده)
5. [عیب‌یابی](#عیب‌یابی)
6. [مثال‌های کاربردی](#مثال‌های-کاربردی)

---

## معرفی

**KuzuDB** یک پایگاه داده گراف embedded و سریع است که برای ذخیره و پردازش گراف دانش (Knowledge Graph) در این پروژه استفاده می‌شود.

### ویژگی‌های کلیدی:
- ✅ **Embedded**: نیازی به سرور جداگانه ندارد
- ✅ **سریع**: بهینه‌سازی شده برای کوئری‌های پیچیده گراف
- ✅ **Cypher**: از زبان کوئری Cypher (مشابه Neo4j) پشتیبانی می‌کند
- ✅ **Python Native**: API ساده و pythonic
- ✅ **ACID**: تراکنش‌های کامل با ضمانت ACID

### نسخه نصب شده:
```
KuzuDB v0.11.3
```

---

## نصب

### پیش‌نیازها:
- Python 3.8 یا بالاتر
- pip یا conda
- سیستم‌عامل: macOS 10.15+, Linux (glibc 2.17+), Windows 10+

### نصب با pip:
```bash
pip install kuzu
```

### نصب نسخه خاص:
```bash
pip install kuzu==0.11.3
```

### بررسی نصب:
```bash
python3 -c "import kuzu; print(f'Kuzu version: {kuzu.__version__}')"
```

خروجی مورد انتظار:
```
Kuzu version: 0.11.3
```

---

## پیکربندی

### 1. استفاده مستقیم از Kuzu

```python
import kuzu
from pathlib import Path

# ایجاد دیتابیس
db_path = Path("./data/kuzu_db")
db_path.mkdir(parents=True, exist_ok=True)

db = kuzu.Database(str(db_path))
conn = kuzu.Connection(db)

# ایجاد schema
conn.execute("""
    CREATE NODE TABLE Entity(
        id STRING,
        name STRING,
        type STRING,
        PRIMARY KEY (id)
    )
""")

# درج داده
conn.execute("CREATE (:Entity {id: 'e1', name: 'Python', type: 'LANGUAGE'})")

# کوئری
result = conn.execute("MATCH (e:Entity) RETURN e.name, e.type")
while result.has_next():
    row = result.get_next()
    print(row)
```

### 2. استفاده از Wrapper (توصیه می‌شود)

```python
from knowledg_graph.kuzu_wrapper import (
    ensure_kuzu_available,
    get_kuzu_client,
    KuzuAvailable
)

# بررسی در دسترس بودن
if not KuzuAvailable:
    print("KuzuDB is not installed!")
    exit(1)

# ایجاد client
client = get_kuzu_client("./data/kuzu_db")

# استفاده
result = client.execute("MATCH (n) RETURN n LIMIT 10")
print(result.data)

# بستن اتصال
client.disconnect()
```

### 3. استفاده از AsyncKuzuManager

```python
from knowledg_graph.kuzudb_package.async_manager import AsyncKuzuManager

async def main():
    async with AsyncKuzuManager("./data/kuzu_db") as kuzu:
        # ایجاد schema
        await kuzu.execute("""
            CREATE NODE TABLE IF NOT EXISTS Entity(
                id STRING PRIMARY KEY,
                name STRING
            )
        """)
        
        # درج داده
        await kuzu.execute(
            "CREATE (:Entity {id: $id, name: $name})",
            parameters={"id": "e1", "name": "Test"}
        )
        
        # کوئری
        result = await kuzu.execute("MATCH (e:Entity) RETURN e")
        print(result.data)

# اجرا
import asyncio
asyncio.run(main())
```

---

## استفاده

### Schema گراف دانش

پروژه از schema زیر استفاده می‌کند:

```cypher
-- Entities (موجودیت‌ها)
CREATE NODE TABLE Entity(
    id STRING PRIMARY KEY,
    name STRING,
    canonical_name STRING,
    type STRING,
    description STRING,
    confidence DOUBLE,
    linked_id STRING,
    kb_url STRING
)

-- Relations (روابط)
CREATE REL TABLE RELATION(
    FROM Entity TO Entity,
    type STRING,
    weight DOUBLE,
    confidence DOUBLE,
    source_doc STRING,
    source_chunk STRING
)

-- Documents (اسناد)
CREATE NODE TABLE Document(
    id STRING PRIMARY KEY,
    title STRING,
    source STRING
)

-- Chunks (بخش‌های متن)
CREATE NODE TABLE Chunk(
    id STRING PRIMARY KEY,
    content STRING,
    doc_id STRING,
    chunk_idx INT64
)

-- Chunk → Entity
CREATE REL TABLE MENTIONS(
    FROM Chunk TO Entity,
    confidence DOUBLE
)

-- Document → Chunk
CREATE REL TABLE HAS_CHUNK(
    FROM Document TO Chunk
)
```

### عملیات پایه

#### 1. درج Entity
```python
await kuzu.execute("""
    CREATE (:Entity {
        id: 'python_lang',
        name: 'Python',
        canonical_name: 'Python Programming Language',
        type: 'TECHNOLOGY',
        description: 'A high-level programming language',
        confidence: 0.95,
        linked_id: 'Q28865',
        kb_url: 'https://www.wikidata.org/wiki/Q28865'
    })
""")
```

#### 2. ایجاد رابطه
```python
await kuzu.execute("""
    MATCH (a:Entity {id: 'python_lang'}),
          (b:Entity {id: 'ml_field'})
    CREATE (a)-[:RELATION {
        type: 'USED_IN',
        weight: 0.8,
        confidence: 0.85,
        source_doc: 'doc123',
        source_chunk: 'chunk456'
    }]->(b)
""")
```

#### 3. کوئری گراف
```python
# پیدا کردن همسایگان
result = await kuzu.execute("""
    MATCH (start:Entity {id: $entity_id})-[r:RELATION*1..2]-(neighbor:Entity)
    RETURN DISTINCT neighbor.id, neighbor.name, neighbor.type
    LIMIT 10
""", parameters={"entity_id": "python_lang"})

for row in result.data:
    print(f"Neighbor: {row['neighbor.name']} ({row['neighbor.type']})")
```

#### 4. مسیریابی
```python
# کوتاه‌ترین مسیر بین دو entity
result = await kuzu.execute("""
    MATCH (a:Entity {id: $start}),
          (b:Entity {id: $end})
    MATCH p = (a)-[:RELATION* SHORTEST 1..5]-(b)
    RETURN nodes(p) AS path_nodes,
           relationships(p) AS path_rels
    LIMIT 1
""", parameters={"start": "e1", "end": "e2"})
```

---

## عیب‌یابی

### مشکل 1: ModuleNotFoundError: No module named 'kuzu'

**علت**: KuzuDB نصب نشده است.

**راه‌حل**:
```bash
pip3 install kuzu
```

### مشکل 2: Runtime exception: Database path cannot be a directory

**علت**: در نسخه‌های قدیمی Kuzu، مسیر فایل می‌خواست. در نسخه 0.11+، مسیر دایرکتوری می‌خواهد.

**راه‌حل**:
```python
# ❌ اشتباه (نسخه قدیم)
db = kuzu.Database("./data/kuzu_db/graph.db")

# ✅ درست (نسخه 0.11+)
db = kuzu.Database("./data/kuzu_db")
```

### مشکل 3: Connection thread safety issues

**علت**: استفاده از یک connection در چند thread.

**راه‌حل**: از `_ConnectionPool` در `kuzu_wrapper.py` استفاده کنید:
```python
from knowledg_graph.kuzu_wrapper import get_kuzu_client

# هر thread یک connection مستقل دریافت می‌کند
client = get_kuzu_client("./data/kuzu_db")
```

### مشکل 4: Schema already exists

**علت**: تلاش برای ایجاد مجدد جدول موجود.

**راه‌حل**: از `IF NOT EXISTS` استفاده کنید:
```python
conn.execute("""
    CREATE NODE TABLE IF NOT EXISTS Entity(...)
""")
```

### مشکل 5: Transaction rollback

**علت**: خطا در میانه تراکنش.

**راه‌حل**: از context manager استفاده کنید:
```python
with client.transaction() as conn:
    conn.execute("CREATE (:Entity {id: 'e1'})")
    conn.execute("CREATE (:Entity {id: 'e2'})")
    # اگر خطا رخ دهد، خودکار rollback می‌شود
```

---

## مثال‌های کاربردی

### مثال 1: ساخت گراف دانش از متن

```python
from knowledg_graph.kuzu_wrapper import get_kuzu_client
from ingestion.EntityExtractionPipeline import EntityExtractionPipeline
from ingestion.relation_pipeline import RelationPipeline
from ingestion.graph_builder import GraphBuilder

# Initialize
client = get_kuzu_client("./data/kuzu_db")
entity_pipeline = EntityExtractionPipeline(config)
relation_pipeline = RelationPipeline(config)
graph_builder = GraphBuilder(client)

# متن ورودی
text = """
Python is a high-level programming language.
It is widely used in machine learning and data science.
"""

# استخراج entities
entities = entity_pipeline.extract(text)
print(f"Found {len(entities)} entities")

# استخراج relations
relations = relation_pipeline.extract(text, entities)
print(f"Found {len(relations)} relations")

# ساخت گراف
await graph_builder.build(
    chunk={"id": "chunk1", "text": text},
    entities=entities,
    relations=relations
)

print("✅ Knowledge graph built successfully!")
```

### مثال 2: جستجو در گراف

```python
from knowledg_graph.kuzu_wrapper import get_kuzu_client

client = get_kuzu_client("./data/kuzu_db")

# جستجوی entity بر اساس نام
result = client.execute("""
    MATCH (e:Entity)
    WHERE e.name CONTAINS $query
    RETURN e.id, e.name, e.type, e.description
    LIMIT 10
""", parameters={"query": "Python"})

for row in result.data:
    print(f"- {row['e.name']} ({row['e.type']}): {row['e.description']}")
```

### مثال 3: تحلیل گراف

```python
# پرارتباط‌ترین entities
result = client.execute("""
    MATCH (e:Entity)-[r]-()
    RETURN e.id, e.name, e.type, count(r) AS degree
    ORDER BY degree DESC
    LIMIT 10
""")

print("Most connected entities:")
for row in result.data:
    print(f"- {row['e.name']}: {row['degree']} connections")

# آمار کلی گراف
stats = client.execute("""
    MATCH (e:Entity) RETURN count(e) AS entity_count
""").first()

relations = client.execute("""
    MATCH ()-[r:RELATION]->() RETURN count(r) AS rel_count
""").first()

print(f"\nGraph Stats:")
print(f"- Entities: {stats['entity_count']}")
print(f"- Relations: {relations['rel_count']}")
```

### مثال 4: بروزرسانی تدریجی

```python
from ingestion.semantic_graph_builder import SemanticGraphBuilder

builder = SemanticGraphBuilder(client)

# بروزرسانی تدریجی (فقط تغییرات)
new_entities = [...]  # entities جدید
new_relations = [...]  # relations جدید

await builder.incremental_update(
    entities=new_entities,
    relations=new_relations,
    version="v2.0"
)

print("✅ Incremental update completed")
```

---

## منابع بیشتر

- 📚 [KuzuDB Documentation](https://kuzudb.com/docs/)
- 🐙 [KuzuDB GitHub](https://github.com/kuzudb/kuzu)
- 💬 [KuzuDB Discord](https://discord.gg/VtX2gw9Rug)
- 📖 [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)

---

## تست سیستم

برای تست کامل یکپارچگی Kuzu:

```bash
# تست wrapper
python3 knowledg_graph/kuzu_wrapper.py

# تست یکپارچگی
python3 tests/test_kuzu_integration.py
```

خروجی مورد انتظار:
```
✅ KuzuDB is AVAILABLE
   Version: 0.11.3
✅ Connection successful!
```

---

## نکات مهم

1. **مسیر دیتابیس**: همیشه مسیر دایرکتوری بدهید، نه فایل
2. **Thread Safety**: از wrapper استفاده کنید تا هر thread connection مستقل داشته باشد
3. **Transactions**: برای عملیات چندگانه از transaction استفاده کنید
4. **Schema**: قبل از درج داده، schema را ایجاد کنید
5. **Cleanup**: بعد از استفاده، connection را ببندید

---

**آخرین بروزرسانی**: 2026-06-07  
**نسخه**: 1.0.0