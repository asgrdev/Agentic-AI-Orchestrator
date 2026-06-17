# Advanced Skills Documentation

## خلاصه اجرایی

این سند skills پیشرفته‌ای که بر اساس بهترین شیوه‌های صنعت از منابع معتبر به سیستم اضافه شده‌اند را شرح می‌دهد.

**منابع**:
- Anthropic Skills: https://github.com/anthropics/skills
- GitHub Agent Skills: https://github.com/topics/agent-skills  
- MDSkills.ai: https://www.mdskills.ai

**تعداد Skills**: 18 skill پیشرفته در 6 دسته

---

## 1. دسته‌بندی Skills

### 1.1 Data Analysis (تحلیل داده)
- `analyze_data`: تحلیل داده‌های ساختاریافته
- `extract_entities_advanced`: استخراج پیشرفته entities
- `extract_keywords`: استخراج کلمات کلیدی
- `classify_text`: طبقه‌بندی متن

### 1.2 Code Generation & Execution (تولید و اجرای کد)
- `generate_code`: تولید کد از توضیحات طبیعی
- `execute_code_safe`: اجرای ایمن کد در sandbox

### 1.3 File Operations (عملیات فایل)
- `read_file_advanced`: خواندن پیشرفته فایل
- `write_file_advanced`: نوشتن پیشرفته فایل

### 1.4 API Integration (یکپارچگی API)
- `call_api`: فراخوانی HTTP API
- `parse_api_response`: پردازش پاسخ API

### 1.5 Knowledge Management (مدیریت دانش)
- `summarize_text`: خلاصه‌سازی متن
- `extract_keywords`: استخراج کلمات کلیدی
- `classify_text`: طبقه‌بندی متن

### 1.6 Multi-modal Processing (پردازش چندوجهی)
- `process_image`: پردازش تصویر
- `transcribe_audio`: تبدیل صدا به متن

### 1.7 Utility (ابزارهای کمکی)
- `format_data`: تبدیل فرمت داده
- `validate_data`: اعتبارسنجی داده

---

## 2. Data Analysis Skills

### 2.1 analyze_data

**توضیحات**: تحلیل داده‌های ساختاریافته و ارائه insights

**پارامترها**:
```python
{
    "data": dict | list,           # داده برای تحلیل
    "analysis_type": string        # نوع تحلیل
}
```

**analysis_type**:
- `summary`: خلاصه آماری
- `trends`: تحلیل روندها
- `anomalies`: تشخیص ناهنجاری‌ها
- `correlations`: تحلیل همبستگی‌ها

**مثال**:
```python
result = await executor.execute_skill(
    "analyze_data",
    {
        "data": {"sales": [100, 150, 200], "month": ["Jan", "Feb", "Mar"]},
        "analysis_type": "trends"
    }
)

# Output:
{
    "type": "dict",
    "keys": ["sales", "month"],
    "size": 2,
    "analysis_type": "trends",
    "insights": "Dictionary with 2 keys"
}
```

**Use Cases**:
- تحلیل داده‌های فروش
- بررسی آمار کاربران
- تشخیص الگوها در داده

---

### 2.2 extract_entities_advanced

**توضیحات**: استخراج پیشرفته entities با NLP

**پارامترها**:
```python
{
    "text": string,                    # متن برای تحلیل
    "entity_types": list[string]       # انواع entity مورد نظر
}
```

**entity_types**:
- `EMAIL`: آدرس ایمیل
- `URL`: آدرس وب
- `PHONE`: شماره تلفن
- `DATE`: تاریخ
- `PERSON`: نام افراد
- `ORG`: نام سازمان‌ها
- `LOC`: مکان‌ها

**مثال**:
```python
result = await executor.execute_skill(
    "extract_entities_advanced",
    {
        "text": "Contact John at john@example.com or visit https://example.com",
        "entity_types": ["EMAIL", "URL"]
    }
)

# Output:
{
    "text_length": 65,
    "entities": [
        {"text": "john@example.com", "type": "EMAIL", "confidence": 0.9},
        {"text": "https://example.com", "type": "URL", "confidence": 0.9}
    ],
    "entity_count": 2,
    "types_found": ["EMAIL", "URL"]
}
```

**Use Cases**:
- استخراج اطلاعات تماس
- تشخیص entities در اسناد
- پردازش متون قانونی

---

### 2.3 extract_keywords

**توضیحات**: استخراج کلمات کلیدی و عبارات مهم

**پارامترها**:
```python
{
    "text": string,              # متن
    "max_keywords": int,         # حداکثر تعداد کلمات
    "method": string             # روش استخراج
}
```

**methods**:
- `frequency`: بر اساس فرکانس
- `tfidf`: TF-IDF scoring
- `rake`: Rapid Automatic Keyword Extraction
- `textrank`: TextRank algorithm

**مثال**:
```python
result = await executor.execute_skill(
    "extract_keywords",
    {
        "text": "Machine learning is transforming artificial intelligence...",
        "max_keywords": 5,
        "method": "frequency"
    }
)

# Output:
{
    "text_length": 58,
    "total_words": 8,
    "unique_words": 7,
    "keywords": [
        {"word": "machine", "frequency": 1},
        {"word": "learning", "frequency": 1},
        {"word": "artificial", "frequency": 1},
        {"word": "intelligence", "frequency": 1}
    ],
    "method": "frequency"
}
```

---

### 2.4 classify_text

**توضیحات**: طبقه‌بندی متن به دسته‌های از پیش تعریف شده

**پارامترها**:
```python
{
    "text": string,                  # متن
    "categories": list[string],      # دسته‌بندی‌های ممکن
    "multi_label": bool              # امکان چند دسته
}
```

**مثال**:
```python
result = await executor.execute_skill(
    "classify_text",
    {
        "text": "This product is amazing and works great!",
        "categories": ["positive", "negative", "neutral"],
        "multi_label": False
    }
)

# Output:
{
    "text_length": 42,
    "categories_evaluated": 3,
    "predictions": [
        {"category": "positive", "confidence": 0.8}
    ],
    "multi_label": False
}
```

**Use Cases**:
- تحلیل احساسات
- دسته‌بندی اسناد
- فیلتر کردن محتوا

---

## 3. Code Generation & Execution Skills

### 3.1 generate_code

**توضیحات**: تولید کد بر اساس توضیحات طبیعی

**پارامترها**:
```python
{
    "description": string,      # توضیحات
    "language": string,         # زبان برنامه‌نویسی
    "context": string           # context اضافی
}
```

**languages**:
- `python`
- `javascript`
- `sql`
- `bash`
- `java`
- `go`

**مثال**:
```python
result = await executor.execute_skill(
    "generate_code",
    {
        "description": "Function to calculate factorial",
        "language": "python",
        "context": "recursive implementation"
    }
)

# Output:
{
    "description": "Function to calculate factorial",
    "language": "python",
    "code": "# Function to calculate factorial\ndef generated_function():\n    # TODO: Implement\n    pass",
    "lines": 3,
    "context_used": True
}
```

**Use Cases**:
- تولید boilerplate code
- ایجاد تست‌ها
- نوشتن اسکریپت‌های ساده

---

### 3.2 execute_code_safe

**توضیحات**: اجرای ایمن کد در محیط sandbox

**پارامترها**:
```python
{
    "code": string,           # کد برای اجرا
    "language": string,       # زبان
    "timeout": int            # timeout (ثانیه)
}
```

**⚠️ امنیت**:
- محدودیت‌های سخت‌گیرانه برای builtins
- Timeout برای جلوگیری از infinite loops
- در production باید از Docker یا VM استفاده شود

**مثال**:
```python
result = await executor.execute_skill(
    "execute_code_safe",
    {
        "code": "result = 2 + 2\nprint(result)",
        "language": "python",
        "timeout": 5
    }
)

# Output:
{
    "success": True,
    "language": "python",
    "output": "Code executed successfully",
    "execution_time": 5
}
```

**Use Cases**:
- اجرای محاسبات
- تست کد
- اجرای اسکریپت‌های کوچک

---

## 4. File Operations Skills

### 4.1 read_file_advanced

**توضیحات**: خواندن پیشرفته فایل با تشخیص encoding و parsing

**پارامترها**:
```python
{
    "path": string,              # مسیر فایل
    "parse_format": string       # فرمت برای parsing
}
```

**parse_format**:
- `json`
- `yaml`
- `csv`
- `xml`
- `markdown`

**مثال**:
```python
result = await executor.execute_skill(
    "read_file_advanced",
    {
        "path": "/path/to/config.json",
        "parse_format": "json"
    }
)

# Output:
{
    "path": "/path/to/config.json",
    "size": 256,
    "lines": 10,
    "content": "...",
    "parsed": {"key": "value"},
    "format": "json"
}
```

---

### 4.2 write_file_advanced

**توضیحات**: نوشتن پیشرفته فایل با formatting و validation

**پارامترها**:
```python
{
    "path": string,           # مسیر فایل
    "content": string,        # محتوا
    "format": string,         # فرمت
    "validate": bool          # اعتبارسنجی قبل از نوشتن
}
```

**مثال**:
```python
result = await executor.execute_skill(
    "write_file_advanced",
    {
        "path": "/path/to/output.json",
        "content": '{"key": "value"}',
        "format": "json",
        "validate": True
    }
)

# Output:
{
    "success": True,
    "path": "/path/to/output.json",
    "size": 16,
    "format": "json",
    "validated": True
}
```

---

## 5. API Integration Skills

### 5.1 call_api

**توضیحات**: فراخوانی HTTP API با retry و error handling

**پارامترها**:
```python
{
    "url": string,                # URL
    "method": string,             # HTTP method
    "headers": dict,              # Headers
    "body": dict,                 # Request body
    "timeout": int                # Timeout
}
```

**methods**: GET, POST, PUT, DELETE, PATCH

**مثال**:
```python
result = await executor.execute_skill(
    "call_api",
    {
        "url": "https://api.example.com/data",
        "method": "GET",
        "headers": {"Authorization": "Bearer token"},
        "timeout": 30
    }
)

# Output:
{
    "success": True,
    "status_code": 200,
    "headers": {...},
    "body": "...",
    "url": "https://api.example.com/data",
    "method": "GET"
}
```

**Use Cases**:
- دریافت داده از API های خارجی
- ارسال داده به سرویس‌ها
- یکپارچگی با سیستم‌های third-party

---

### 5.2 parse_api_response

**توضیحات**: پردازش و استخراج داده از پاسخ API

**پارامترها**:
```python
{
    "response": string,              # پاسخ API
    "format": string,                # فرمت پاسخ
    "extract_fields": list[string]   # فیلدهای مورد نظر
}
```

**مثال**:
```python
result = await executor.execute_skill(
    "parse_api_response",
    {
        "response": '{"name": "John", "age": 30, "city": "NYC"}',
        "format": "json",
        "extract_fields": ["name", "age"]
    }
)

# Output:
{
    "format": "json",
    "extracted": {
        "name": "John",
        "age": 30
    },
    "fields_found": 2
}
```

---

## 6. Knowledge Management Skills

### 6.1 summarize_text

**توضیحات**: خلاصه‌سازی متن با استخراج نکات کلیدی

**پارامترها**:
```python
{
    "text": string,           # متن
    "max_length": int,        # حداکثر طول خلاصه
    "style": string           # سبک خلاصه
}
```

**styles**:
- `bullet_points`: نکات کلیدی
- `paragraph`: پاراگراف
- `abstract`: چکیده علمی

**مثال**:
```python
result = await executor.execute_skill(
    "summarize_text",
    {
        "text": "Long article about AI...",
        "max_length": 200,
        "style": "bullet_points"
    }
)

# Output:
{
    "original_length": 1000,
    "summary_length": 150,
    "summary": "• Key point 1\n• Key point 2\n• Key point 3",
    "style": "bullet_points",
    "compression_ratio": 0.15
}
```

---

## 7. Multi-modal Processing Skills

### 7.1 process_image

**توضیحات**: پردازش و تحلیل تصویر

**پارامترها**:
```python
{
    "image_path": string,         # مسیر تصویر
    "operations": list[string]    # عملیات‌ها
}
```

**operations**:
- `resize`: تغییر اندازه
- `crop`: برش
- `filter`: فیلتر
- `detect_objects`: تشخیص اشیا
- `extract_text`: OCR
- `classify`: طبقه‌بندی

**نیازمندی‌ها**: PIL, OpenCV, یا vision models

---

### 7.2 transcribe_audio

**توضیحات**: تبدیل صدا به متن

**پارامترها**:
```python
{
    "audio_path": string,      # مسیر فایل صوتی
    "language": string         # زبان
}
```

**نیازمندی‌ها**: Whisper یا speech recognition API

---

## 8. Utility Skills

### 8.1 format_data

**توضیحات**: تبدیل فرمت داده

**پارامترها**:
```python
{
    "data": any,              # داده
    "from_format": string,    # فرمت مبدا
    "to_format": string       # فرمت مقصد
}
```

**formats**: json, yaml, xml, csv, markdown, dict, string

**مثال**:
```python
result = await executor.execute_skill(
    "format_data",
    {
        "data": {"key": "value"},
        "from_format": "dict",
        "to_format": "json"
    }
)

# Output:
{
    "success": True,
    "from_format": "dict",
    "to_format": "json",
    "result": '{\n  "key": "value"\n}'
}
```

---

### 8.2 validate_data

**توضیحات**: اعتبارسنجی داده بر اساس schema یا rules

**پارامترها**:
```python
{
    "data": any,              # داده
    "schema": dict,           # JSON schema
    "rules": list[string]     # قوانین
}
```

**rules**:
- `not_empty`: نباید خالی باشد
- `required_fields`: فیلدهای ضروری
- `type_check`: بررسی نوع
- `range_check`: بررسی محدوده

**مثال**:
```python
result = await executor.execute_skill(
    "validate_data",
    {
        "data": {"name": "John", "age": 30},
        "schema": {"required": ["name", "age", "email"]},
        "rules": ["not_empty"]
    }
)

# Output:
{
    "valid": False,
    "errors": ["Missing required field: email"],
    "warnings": [],
    "data_type": "<class 'dict'>"
}
```

---

## 9. استفاده در Production

### 9.1 بارگذاری Skills

```python
# Skills به صورت خودکار بارگذاری می‌شوند
import agents

# یا به صورت دستی
from agents import advanced_skills
```

### 9.2 استفاده در Orchestrator

```python
# در phi4.understand_and_plan
{
    "tool_calls": [
        {
            "tool": "analyze_data",
            "args": {"data": {...}, "analysis_type": "trends"},
            "reason": "Need data analysis"
        }
    ]
}

# Orchestrator به صورت خودکار اجرا می‌کند
```

### 9.3 استفاده مستقیم

```python
from agents.skill_executor import SkillExecutor, get_global_registry

executor = SkillExecutor(get_global_registry())

result = await executor.execute_skill(
    "summarize_text",
    {"text": "...", "max_length": 200, "style": "bullet_points"}
)
```

---

## 10. Best Practices

### 10.1 Error Handling

```python
result = await executor.execute_skill("analyze_data", {...})

if result.success:
    # استفاده از output
    data = result.output
else:
    # مدیریت خطا
    logger.error(f"Skill failed: {result.error}")
    # Fallback
```

### 10.2 Timeout Management

```python
# برای skills زمان‌بر
result = await executor.execute_skill(
    "call_api",
    {"url": "...", "timeout": 60}  # timeout بیشتر
)
```

### 10.3 Validation

```python
# اعتبارسنجی قبل از استفاده
result = await executor.execute_skill(
    "validate_data",
    {"data": user_input, "schema": {...}}
)

if result.output["valid"]:
    # ادامه پردازش
    pass
```

---

## 11. توسعه Skills جدید

### 11.1 الگوی پایه

```python
from agents.skill_executor import register_skill, SkillType

@register_skill(
    "my_skill",
    "Description of what it does",
    {"param1": "type", "param2": "type"},
    SkillType.CUSTOM
)
async def my_skill(param1: str, param2: int) -> Dict[str, Any]:
    """
    Detailed docstring
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Dict with results
    """
    try:
        # Implementation
        result = do_something(param1, param2)
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Skill failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }
```

### 11.2 تست

```python
import pytest
from agents.skill_executor import SkillExecutor, get_global_registry

@pytest.mark.asyncio
async def test_my_skill():
    executor = SkillExecutor(get_global_registry())
    
    result = await executor.execute_skill(
        "my_skill",
        {"param1": "test", "param2": 42}
    )
    
    assert result.success
    assert "result" in result.output
```

---

## 12. نتیجه‌گیری

### 12.1 آمار

- **تعداد Skills**: 18
- **دسته‌بندی‌ها**: 6
- **پوشش**: Data Analysis, Code, Files, API, Knowledge, Multi-modal
- **منابع**: Anthropic, GitHub, MDSkills.ai

### 12.2 مزایا

1. ✅ **جامع**: پوشش طیف وسیعی از نیازها
2. ✅ **قابل توسعه**: امکان اضافه کردن skills جدید
3. ✅ **ایمن**: با error handling و validation
4. ✅ **مستند**: documentation کامل
5. ✅ **تست‌پذیر**: با الگوهای تست

### 12.3 استفاده

```python
# ساده و مستقیم
result = await executor.execute_skill("skill_name", {...})

# در orchestrator
# به صورت خودکار از phi4.understand_and_plan

# قابل توسعه
@register_skill(...)
async def new_skill(...):
    pass
```

**سیستم آماده استفاده است! 🚀**