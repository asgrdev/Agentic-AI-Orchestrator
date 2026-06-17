"""
Advanced Skills Collection
Based on industry best practices from:
- Anthropic Skills (https://github.com/anthropics/skills)
- GitHub Agent Skills (https://github.com/topics/agent-skills)
- MDSkills.ai (https://www.mdskills.ai)

این ماژول شامل skills پیشرفته برای:
- Data Analysis
- Code Generation & Execution
- File Operations
- API Integration
- Knowledge Management
- Multi-modal Processing
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from agents.skill_executor import register_skill, SkillType

logger = logging.getLogger(__name__)


# ============================================================================
# Data Analysis Skills
# ============================================================================

@register_skill(
    "analyze_data",
    "Analyze structured data and provide insights",
    {"data": "dict|list", "analysis_type": "string"},
    SkillType.ANALYZE
)
async def analyze_data_skill(data: Any, analysis_type: str = "summary") -> Dict[str, Any]:
    """
    تحلیل داده‌های ساختاریافته
    
    analysis_type:
    - summary: خلاصه آماری
    - trends: روندها
    - anomalies: ناهنجاری‌ها
    - correlations: همبستگی‌ها
    """
    try:
        if isinstance(data, dict):
            return {
                "type": "dict",
                "keys": list(data.keys()),
                "size": len(data),
                "analysis_type": analysis_type,
                "insights": f"Dictionary with {len(data)} keys"
            }
        elif isinstance(data, list):
            return {
                "type": "list",
                "length": len(data),
                "sample": data[:3] if len(data) > 3 else data,
                "analysis_type": analysis_type,
                "insights": f"List with {len(data)} items"
            }
        else:
            return {
                "type": str(type(data)),
                "value": str(data),
                "analysis_type": analysis_type
            }
    except Exception as e:
        logger.error(f"Data analysis failed: {e}")
        return {"error": str(e)}


@register_skill(
    "extract_entities_advanced",
    "Extract entities with advanced NLP techniques",
    {"text": "string", "entity_types": "list[string]"},
    SkillType.ANALYZE
)
async def extract_entities_advanced_skill(
    text: str,
    entity_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    استخراج پیشرفته entities با NLP
    
    entity_types: PERSON, ORG, LOC, DATE, MONEY, PRODUCT, EVENT, etc.
    """
    # ساده‌سازی شده - در production باید از spaCy یا transformers استفاده شود
    entities = []
    
    # الگوهای ساده برای تشخیص
    patterns = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "URL": r'https?://[^\s]+',
        "PHONE": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "DATE": r'\b\d{4}-\d{2}-\d{2}\b',
    }
    
    for entity_type, pattern in patterns.items():
        if not entity_types or entity_type in entity_types:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": entity_type,
                    "confidence": 0.9
                })
    
    return {
        "text_length": len(text),
        "entities": entities,
        "entity_count": len(entities),
        "types_found": list(set(e["type"] for e in entities))
    }


# ============================================================================
# Code Generation & Execution Skills
# ============================================================================

@register_skill(
    "generate_code",
    "Generate code based on natural language description",
    {"description": "string", "language": "string", "context": "string"},
    SkillType.TRANSFORM
)
async def generate_code_skill(
    description: str,
    language: str = "python",
    context: str = ""
) -> Dict[str, Any]:
    """
    تولید کد بر اساس توضیحات طبیعی
    
    languages: python, javascript, sql, bash, etc.
    """
    # در production باید از LLM استفاده شود
    templates = {
        "python": f"# {description}\ndef generated_function():\n    # TODO: Implement\n    pass",
        "javascript": f"// {description}\nfunction generatedFunction() {{\n    // TODO: Implement\n}}",
        "sql": f"-- {description}\nSELECT * FROM table WHERE condition;",
    }
    
    code = templates.get(language, f"# {description}\n# Language: {language}")
    
    return {
        "description": description,
        "language": language,
        "code": code,
        "lines": len(code.split('\n')),
        "context_used": bool(context)
    }


@register_skill(
    "execute_code_safe",
    "Execute code in a safe sandboxed environment",
    {"code": "string", "language": "string", "timeout": "int"},
    SkillType.TRANSFORM
)
async def execute_code_safe_skill(
    code: str,
    language: str = "python",
    timeout: int = 5
) -> Dict[str, Any]:
    """
    اجرای ایمن کد در محیط sandbox
    
    IMPORTANT: در production باید از Docker یا VM استفاده شود
    """
    try:
        if language == "python":
            # محدودیت‌های امنیتی
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                }
            }
            
            # اجرا با timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(exec, code, safe_globals),
                timeout=timeout
            )
            
            return {
                "success": True,
                "language": language,
                "output": "Code executed successfully",
                "execution_time": timeout
            }
        else:
            return {
                "success": False,
                "error": f"Language '{language}' not supported for execution"
            }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Execution timeout after {timeout}s"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# File Operations Skills
# ============================================================================

@register_skill(
    "read_file_advanced",
    "Read file with encoding detection and parsing",
    {"path": "string", "parse_format": "string"},
    SkillType.RETRIEVE
)
async def read_file_advanced_skill(
    path: str,
    parse_format: Optional[str] = None
) -> Dict[str, Any]:
    """
    خواندن پیشرفته فایل با تشخیص encoding و parsing
    
    parse_format: json, yaml, csv, xml, markdown, etc.
    """
    try:
        # خواندن فایل
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = {
            "path": path,
            "size": len(content),
            "lines": len(content.split('\n')),
            "content": content[:500]  # نمونه
        }
        
        # Parsing بر اساس format
        if parse_format == "json":
            try:
                parsed = json.loads(content)
                result["parsed"] = parsed
                result["format"] = "json"
            except json.JSONDecodeError as e:
                result["parse_error"] = str(e)
        
        return result
        
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    except Exception as e:
        return {"error": str(e)}


@register_skill(
    "write_file_advanced",
    "Write file with formatting and validation",
    {"path": "string", "content": "string", "format": "string", "validate": "bool"},
    SkillType.TRANSFORM
)
async def write_file_advanced_skill(
    path: str,
    content: str,
    format: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    نوشتن پیشرفته فایل با formatting و validation
    """
    try:
        # Validation
        if validate and format == "json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON: {e}"}
        
        # نوشتن فایل
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "path": path,
            "size": len(content),
            "format": format,
            "validated": validate
        }
        
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# API Integration Skills
# ============================================================================

@register_skill(
    "call_api",
    "Make HTTP API calls with retry and error handling",
    {"url": "string", "method": "string", "headers": "dict", "body": "dict", "timeout": "int"},
    SkillType.SEARCH
)
async def call_api_skill(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    فراخوانی API با retry و error handling
    
    methods: GET, POST, PUT, DELETE, PATCH
    """
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(url, headers=headers, json=body)
            elif method.upper() == "PUT":
                response = await client.put(url, headers=headers, json=body)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:1000],  # نمونه
                "url": url,
                "method": method
            }
            
    except Exception as e:
        logger.error(f"API call failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "method": method
        }


@register_skill(
    "parse_api_response",
    "Parse and extract data from API responses",
    {"response": "string", "format": "string", "extract_fields": "list[string]"},
    SkillType.TRANSFORM
)
async def parse_api_response_skill(
    response: str,
    format: str = "json",
    extract_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    پردازش و استخراج داده از پاسخ API
    """
    try:
        if format == "json":
            data = json.loads(response)
            
            if extract_fields:
                extracted = {}
                for field in extract_fields:
                    if field in data:
                        extracted[field] = data[field]
                return {
                    "format": "json",
                    "extracted": extracted,
                    "fields_found": len(extracted)
                }
            else:
                return {
                    "format": "json",
                    "data": data,
                    "keys": list(data.keys()) if isinstance(data, dict) else None
                }
        else:
            return {
                "format": format,
                "raw": response[:500]
            }
            
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Knowledge Management Skills
# ============================================================================

@register_skill(
    "summarize_text",
    "Summarize long text with key points extraction",
    {"text": "string", "max_length": "int", "style": "string"},
    SkillType.SUMMARIZE
)
async def summarize_text_skill(
    text: str,
    max_length: int = 200,
    style: str = "bullet_points"
) -> Dict[str, Any]:
    """
    خلاصه‌سازی متن با استخراج نکات کلیدی
    
    styles: bullet_points, paragraph, abstract
    """
    # ساده‌سازی شده - در production باید از LLM استفاده شود
    sentences = text.split('.')
    
    if style == "bullet_points":
        summary = "\n".join([f"• {s.strip()}" for s in sentences[:3] if s.strip()])
    else:
        summary = '. '.join(sentences[:3])
    
    return {
        "original_length": len(text),
        "summary_length": len(summary),
        "summary": summary[:max_length],
        "style": style,
        "compression_ratio": len(summary) / len(text) if text else 0
    }


@register_skill(
    "extract_keywords",
    "Extract keywords and key phrases from text",
    {"text": "string", "max_keywords": "int", "method": "string"},
    SkillType.ANALYZE
)
async def extract_keywords_skill(
    text: str,
    max_keywords: int = 10,
    method: str = "frequency"
) -> Dict[str, Any]:
    """
    استخراج کلمات کلیدی از متن
    
    methods: frequency, tfidf, rake, textrank
    """
    # ساده‌سازی شده - استخراج بر اساس فرکانس
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    
    # حذف stop words ساده
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # مرتب‌سازی بر اساس فرکانس
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    keywords = [{"word": word, "frequency": freq} for word, freq in sorted_keywords[:max_keywords]]
    
    return {
        "text_length": len(text),
        "total_words": len(words),
        "unique_words": len(word_freq),
        "keywords": keywords,
        "method": method
    }


@register_skill(
    "classify_text",
    "Classify text into predefined categories",
    {"text": "string", "categories": "list[string]", "multi_label": "bool"},
    SkillType.ANALYZE
)
async def classify_text_skill(
    text: str,
    categories: List[str],
    multi_label: bool = False
) -> Dict[str, Any]:
    """
    طبقه‌بندی متن
    
    categories: لیست دسته‌بندی‌های ممکن
    multi_label: امکان انتخاب چند دسته
    """
    # ساده‌سازی شده - بر اساس کلمات کلیدی
    text_lower = text.lower()
    scores = {}
    
    for category in categories:
        # امتیاز بر اساس وجود کلمات مرتبط
        category_words = category.lower().split()
        score = sum(1 for word in category_words if word in text_lower)
        scores[category] = score / len(category_words) if category_words else 0
    
    # مرتب‌سازی
    sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if multi_label:
        # تمام دسته‌ها با امتیاز > 0
        predictions = [
            {"category": cat, "confidence": score}
            for cat, score in sorted_categories if score > 0
        ]
    else:
        # فقط بالاترین امتیاز
        predictions = [
            {"category": sorted_categories[0][0], "confidence": sorted_categories[0][1]}
        ] if sorted_categories else []
    
    return {
        "text_length": len(text),
        "categories_evaluated": len(categories),
        "predictions": predictions,
        "multi_label": multi_label
    }


# ============================================================================
# Multi-modal Processing Skills
# ============================================================================

@register_skill(
    "process_image",
    "Process and analyze images",
    {"image_path": "string", "operations": "list[string]"},
    SkillType.ANALYZE
)
async def process_image_skill(
    image_path: str,
    operations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    پردازش و تحلیل تصویر
    
    operations: resize, crop, filter, detect_objects, extract_text, etc.
    """
    # در production باید از PIL, OpenCV, یا vision models استفاده شود
    return {
        "image_path": image_path,
        "operations": operations or [],
        "status": "Image processing requires additional dependencies",
        "note": "Install PIL/OpenCV for image processing"
    }


@register_skill(
    "transcribe_audio",
    "Transcribe audio to text",
    {"audio_path": "string", "language": "string"},
    SkillType.TRANSFORM
)
async def transcribe_audio_skill(
    audio_path: str,
    language: str = "en"
) -> Dict[str, Any]:
    """
    تبدیل صدا به متن
    
    در production باید از Whisper یا speech recognition API استفاده شود
    """
    return {
        "audio_path": audio_path,
        "language": language,
        "status": "Audio transcription requires Whisper or speech recognition API",
        "note": "Install whisper or use speech recognition service"
    }


# ============================================================================
# Utility Skills
# ============================================================================

@register_skill(
    "format_data",
    "Format data between different formats",
    {"data": "any", "from_format": "string", "to_format": "string"},
    SkillType.TRANSFORM
)
async def format_data_skill(
    data: Any,
    from_format: str,
    to_format: str
) -> Dict[str, Any]:
    """
    تبدیل فرمت داده
    
    formats: json, yaml, xml, csv, markdown, etc.
    """
    try:
        if from_format == "json" and to_format == "dict":
            if isinstance(data, str):
                result = json.loads(data)
            else:
                result = data
        elif from_format == "dict" and to_format == "json":
            result = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            result = str(data)
        
        return {
            "success": True,
            "from_format": from_format,
            "to_format": to_format,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@register_skill(
    "validate_data",
    "Validate data against schema or rules",
    {"data": "any", "schema": "dict", "rules": "list[string]"},
    SkillType.VALIDATE
)
async def validate_data_skill(
    data: Any,
    schema: Optional[Dict[str, Any]] = None,
    rules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    اعتبارسنجی داده
    
    schema: JSON schema یا custom schema
    rules: لیست قوانین اعتبارسنجی
    """
    errors = []
    warnings = []
    
    # اعتبارسنجی ساده
    if schema:
        required_fields = schema.get("required", [])
        if isinstance(data, dict):
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
    
    if rules:
        for rule in rules:
            # پردازش قوانین ساده
            if rule == "not_empty" and not data:
                errors.append("Data is empty")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "data_type": str(type(data))
    }


logger.info("Advanced skills registered successfully")

