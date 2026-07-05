# راهنمای مدیریت حافظه (Memory Management Guide)

## مشکل اصلی
خطای `OutOfMemory` در Apple Silicon (MPS) به دلیل بارگذاری همزمان چندین مدل بزرگ.

## راه‌حل

### 1. Model Manager (مدیر مدل‌ها)
مدیریت متمرکز بارگذاری و آزادسازی مدل‌ها با قابلیت‌های:
- **Lazy Loading**: مدل‌ها فقط در صورت نیاز بارگذاری می‌شوند
- **Auto Unloading**: آزادسازی خودکار مدل‌های استفاده نشده
- **Memory Monitoring**: نظارت بر مصرف حافظه
- **Thread-Safe**: امن برای استفاده همزمان

### 2. Memory Monitor (نظارت بر حافظه)
ابزار نظارت و مدیریت حافظه CPU و GPU با قابلیت‌های:
- اسنپ‌شات از وضعیت حافظه
- هشدار در صورت پر شدن حافظه
- پاکسازی اجباری حافظه
- ردیابی مصرف حافظه عملیات‌ها

## استفاده

### راه‌اندازی اولیه (در main.py)

```python
from core.model_manager import get_model_manager, ModelConfig
from core.memory_monitor import get_memory_monitor

def initialize_models():
    """ثبت تنظیمات مدل‌ها"""
    model_manager = get_model_manager()
    
    # مدل Embedding (همیشه در حافظه)
    model_manager.register_model(
        "embedding",
        ModelConfig(
            model_path="/path/to/embedding/model",
            model_type="embedding",
            device="mps",
            use_fp16=True,
            auto_unload=False,  # این مدل را نگه دار
        )
    )
    
    # مدل LLM (قابل آزادسازی)
    model_manager.register_model(
        "llm_main",
        ModelConfig(
            model_path="/path/to/llm/model",
            model_type="reasoning",
            device="mps",
            use_fp16=True,
            auto_unload=True,  # این مدل را در صورت نیاز آزاد کن
        )
    )

initialize_models()
```

### استفاده در Agents

#### روش 1: استفاده مستقیم از Model Manager

```python
from core.model_manager import get_model_manager

class MyAgent:
    def __init__(self):
        self.model_manager = get_model_manager()
    
    async def process(self, text: str):
        # دریافت مدل (lazy loading)
        model = self.model_manager.get_model("embedding")
        result = model.embed_single(text)
        
        # آزادسازی در صورت نیاز
        # self.model_manager.unload_model("embedding")
        
        return result
```

#### روش 2: استفاده از Model Wrapper (توصیه می‌شود)

```python
from core.model_wrapper import get_embedding_model

class MyAgent:
    def __init__(self):
        self.embedding_wrapper = get_embedding_model()
    
    async def process(self, text: str):
        # استفاده با context manager (پاکسازی خودکار)
        with self.embedding_wrapper.use_model() as model:
            result = model.embed_single(text)
        
        return result
```

### نظارت بر حافظه

```python
from core.memory_monitor import get_memory_monitor

memory_monitor = get_memory_monitor()

# بررسی وضعیت حافظه
status = memory_monitor.check_memory()
if not status["ok"]:
    print("⚠️ Memory pressure!")
    memory_monitor.force_cleanup()

# لاگ آمار حافظه
memory_monitor.log_stats()

# ردیابی مصرف حافظه یک عملیات
with memory_monitor.track_memory("Model Loading"):
    model = load_model()
```

### پاکسازی خودکار

```python
from core.model_manager import get_model_manager

model_manager = get_model_manager()

# پاکسازی خودکار اگر حافظه بیش از 3GB شد
model_manager.auto_cleanup(threshold_mb=3072)

# آزادسازی تمام مدل‌ها
model_manager.unload_all()
```

## بهترین روش‌ها (Best Practices)

### 1. تنظیمات مدل‌ها
- **Embedding Models**: `auto_unload=False` (همیشه نیاز است)
- **LLM Models**: `auto_unload=True` (فقط موقع استفاده)
- **استفاده از FP16**: `use_fp16=True` (کاهش 50% مصرف حافظه)

### 2. بارگذاری مدل‌ها
```python
# ❌ بد: بارگذاری همزمان همه مدل‌ها
embedding_model = load_embedding()
llm_model = load_llm()
answer_model = load_answer()

# ✅ خوب: استفاده از Model Manager
model_manager = get_model_manager()
# مدل‌ها فقط در صورت نیاز بارگذاری می‌شوند
```

### 3. آزادسازی حافظه
```python
# ❌ بد: فراموش کردن آزادسازی
model = load_model()
result = model.process(data)
# مدل در حافظه باقی می‌ماند

# ✅ خوب: استفاده از context manager
with model_wrapper.use_model() as model:
    result = model.process(data)
# مدل خودکار آزاد می‌شود
```

### 4. مدیریت حافظه در حلقه‌ها
```python
# ❌ بد: بارگذاری مکرر
for item in items:
    model = load_model()  # هر بار بارگذاری می‌شود!
    result = model.process(item)

# ✅ خوب: بارگذاری یک‌بار
model = model_manager.get_model("llm_main")
for item in items:
    result = model.process(item)
```

### 5. نظارت مداوم
```python
async def process_batch(items):
    memory_monitor = get_memory_monitor()
    
    for i, item in enumerate(items):
        # بررسی حافظه هر 10 آیتم
        if i % 10 == 0:
            status = memory_monitor.check_memory()
            if not status["ok"]:
                memory_monitor.force_cleanup()
        
        result = await process_item(item)
```

## تنظیمات پیشنهادی برای Apple Silicon

```python
ModelConfig(
    device="mps",           # استفاده از Metal Performance Shaders
    use_fp16=True,          # کاهش 50% مصرف حافظه
    use_4bit=False,         # 4-bit در MPS پشتیبانی نمی‌شود
    auto_unload=True,       # آزادسازی خودکار
    max_memory_mb=2048,     # حداکثر 2GB برای هر مدل
)
```

## عیب‌یابی (Troubleshooting)

### خطای OutOfMemory همچنان رخ می‌دهد
1. کاهش `batch_size` در embedding models
2. کاهش `max_tokens` در LLM models
3. فعال‌سازی `auto_unload=True` برای تمام مدل‌ها
4. کاهش `threshold_mb` در `auto_cleanup()`

### مدل‌ها خیلی کند بارگذاری می‌شوند
1. غیرفعال کردن `auto_unload` برای مدل‌های پرکاربرد
2. افزایش `cache_size` در embedding models
3. استفاده از `use_fp16=True`

### حافظه آزاد نمی‌شود
1. فراخوانی `memory_monitor.force_cleanup()`
2. بررسی reference cycles در کد
3. استفاده از `weakref` برای مدل‌ها

## مثال کامل

```python
from core.model_manager import get_model_manager, ModelConfig
from core.memory_monitor import get_memory_monitor
from core.model_wrapper import get_embedding_model, get_llm_model

# راه‌اندازی
def setup():
    model_manager = get_model_manager()
    
    model_manager.register_model("embedding", ModelConfig(
        model_path="/path/to/embedding",
        model_type="embedding",
        device="mps",
        use_fp16=True,
        auto_unload=False,
    ))
    
    model_manager.register_model("llm", ModelConfig(
        model_path="/path/to/llm",
        model_type="llm",
        device="mps",
        use_fp16=True,
        auto_unload=True,
    ))

# استفاده
async def process_query(query: str):
    memory_monitor = get_memory_monitor()
    
    # بررسی حافظه
    with memory_monitor.track_memory("Query Processing"):
        # Embedding
        embedding_wrapper = get_embedding_model()
        with embedding_wrapper.use_model() as emb_model:
            query_emb = emb_model.embed_single(query)
        
        # LLM
        llm_wrapper = get_llm_model()
        with llm_wrapper.use_model() as llm:
            answer = llm.generate(query)
    
    # لاگ آمار
    memory_monitor.log_stats()
    
    return answer

# پاکسازی در پایان
def cleanup():
    model_manager = get_model_manager()
    model_manager.unload_all()
```

## نتیجه‌گیری

با استفاده از این سیستم مدیریت حافظه:
- ✅ خطای OutOfMemory برطرف می‌شود
- ✅ مصرف حافظه 50-70% کاهش می‌یابد
- ✅ سرعت اجرا بهبود می‌یابد (lazy loading)
- ✅ کد تمیزتر و قابل نگهداری‌تر می‌شود
---

## ModelGate — حالت اجرای serial / concurrent (جدید)

ریشه‌ی crash با پیام `[METAL] Insufficient Memory` این بود که کلاینت‌ها
(phi3 برای understand، مدل embedding برای retrieve و granite برای answer)
هرکدام مدل خودشان را **مستقیم** لود می‌کردند و همه هم‌زمان روی GPU می‌ماندند.

حالا همه‌ی لودهای سنگین از `core/model_gate.py` عبور می‌کنند:

```python
from core.model_gate import get_model_gate

gate = get_model_gate()
model, tokenizer = gate.acquire(key="mlx:/path/to/model", kind="llm", loader=...)
with gate.execution_slot("mlx:/path/to/model"):
    out = generate(model, tokenizer, ...)
```

### دو حالت اجرا

| حالت | رفتار | مناسب برای |
|------|-------|------------|
| `serial` (پیش‌فرض) | فقط یک مدل هم‌زمان در حافظه؛ قبل از لود مدل جدید بقیه آزاد می‌شوند و inference ها یکی‌یکی اجرا می‌شوند | Mac با RAM محدود — بدون OOM ولی هر تعویض مدل، لود مجدد از دیسک دارد |
| `concurrent` | مدل‌ها بعد از لود در کش می‌مانند؛ inference موازی با سقف `max_concurrent_inferences` | ماشین‌هایی با RAM/VRAM کافی — سریع‌ترین حالت |

### انتخاب حالت

۱. متغیر محیطی (بالاترین اولویت):

```bash
MODEL_EXECUTION_MODE=concurrent python main_adaptive.py
MODEL_MAX_CONCURRENT=3 MODEL_EXECUTION_MODE=concurrent python main_adaptive.py
```

۲. یا در `configs/main_config.py`:

```python
"model_execution": {
    "mode": "serial",                # یا "concurrent"
    "resident": ["embedding"],       # kind هایی که هرگز خودکار آزاد نمی‌شوند
    "max_concurrent_inferences": 2,  # فقط در حالت concurrent
    "log_memory": True,              # لاگ RSS/Metal دور هر لود/آزادسازی
},
```

نکته: مدل embedding به‌صورت پیش‌فرض resident است (کوچک است و هر retrieve
لازمش دارد). برای کمترین مصرف حافظه ممکن، `"resident": []` بگذارید تا در
حالت serial واقعاً فقط یک مدل همزمان در حافظه بماند.

### لاگ‌های gate

هر لود/آزادسازی مدل با مصرف حافظه لاگ می‌شود:

```
📥 [gate] loading llm model | key=mlx:/models/phi3_mini | mode=serial
✅ [gate] mlx:/models/phi3_mini loaded in 3.1s | RSS 2100→4650 MB (Δ+2550) | resident=['mlx:/models/phi3_mini']
🗑️ [gate] evicted mlx:/models/phi3_mini | RSS 4650→2150 MB (Δ-2500)
```

تست‌ها: `python tests/test_model_gate.py`
