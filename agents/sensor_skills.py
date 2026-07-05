"""
Sensor Skills for Skill Registry
=================================

Skills برای کار با سنسورها که با SkillExecutor و phi4 یکپارچه می‌شوند.

Skills:
1. Vision Skills: object detection, scene classification
2. Audio Skills: transcription, audio event detection  
3. Text Skills: embedding, language detection
4. Multi-modal Skills: combined processing
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# VISION SKILLS
# ═══════════════════════════════════════════════════════════════

def detect_objects_in_image(image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
    """
    تشخیص اشیاء در تصویر با YOLO
    
    Args:
        image_path: مسیر تصویر
        conf_threshold: آستانه اطمینان (0-1)
    
    Returns:
        dict با لیست اشیاء تشخیص داده شده
    
    Example:
        >>> result = detect_objects_in_image("photo.jpg", conf_threshold=0.3)
        >>> print(f"Found {result['count']} objects")
    """
    try:
        from agents.sensor_integration_system import VisionSensorWrapper
        
        vision = VisionSensorWrapper()
        result = vision.detect_objects(image_path, conf_threshold)
        
        return {
            "success": result.get("success", False),
            "objects": result.get("detections", []),
            "count": result.get("count", 0),
            "image_path": image_path,
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "objects": [],
            "count": 0
        }


def classify_image_scene(image_path: str) -> Dict[str, Any]:
    """
    طبقه‌بندی صحنه تصویر
    
    Args:
        image_path: مسیر تصویر
    
    Returns:
        dict با نوع صحنه و اطمینان
    
    Example:
        >>> result = classify_image_scene("landscape.jpg")
        >>> print(f"Scene: {result['scene']} ({result['confidence']:.2f})")
    """
    try:
        from agents.sensor_integration_system import VisionSensorWrapper
        
        vision = VisionSensorWrapper()
        result = vision.classify_scene(image_path)
        
        return {
            "success": result.get("success", False),
            "scene": result.get("scene", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "image_path": image_path,
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"Scene classification failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "scene": "unknown",
            "confidence": 0.0
        }


# ═══════════════════════════════════════════════════════════════
# AUDIO SKILLS
# ═══════════════════════════════════════════════════════════════

def transcribe_audio_file(audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    تبدیل گفتار به متن با Whisper
    
    Args:
        audio_path: مسیر فایل صوتی
        language: زبان (auto-detect if None)
    
    Returns:
        dict با متن و زبان تشخیص داده شده
    
    Example:
        >>> result = transcribe_audio_file("speech.wav", language="en")
        >>> print(f"Text: {result['text']}")
    """
    try:
        from agents.sensor_integration_system import AudioSensorWrapper
        
        audio = AudioSensorWrapper()
        result = audio.transcribe(audio_path, language)
        
        return {
            "success": result.get("success", False),
            "text": result.get("text", ""),
            "language": result.get("language", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "audio_path": audio_path,
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "text": "",
            "language": "unknown"
        }


def detect_audio_events(audio_path: str) -> Dict[str, Any]:
    """
    تشخیص رویدادهای صوتی
    
    Args:
        audio_path: مسیر فایل صوتی
    
    Returns:
        dict با لیست رویدادهای تشخیص داده شده
    
    Example:
        >>> result = detect_audio_events("sounds.wav")
        >>> print(f"Found {result['count']} events")
    """
    try:
        from agents.sensor_integration_system import AudioSensorWrapper
        
        audio = AudioSensorWrapper()
        result = audio.detect_audio_events(audio_path)
        
        return {
            "success": result.get("success", False),
            "events": result.get("events", []),
            "count": result.get("count", 0),
            "audio_path": audio_path,
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"Audio event detection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "events": [],
            "count": 0
        }


# ═══════════════════════════════════════════════════════════════
# TEXT SKILLS
# ═══════════════════════════════════════════════════════════════

def embed_text_with_bert(text: str) -> Dict[str, Any]:
    """
    تولید embedding برای متن با BERT
    
    Args:
        text: متن ورودی
    
    Returns:
        dict با embedding vector
    
    Example:
        >>> result = embed_text_with_bert("Hello world")
        >>> print(f"Embedding dimension: {result['dimension']}")
    """
    try:
        from agents.sensor_integration_system import TextSensorWrapper
        
        text_sensor = TextSensorWrapper()
        result = text_sensor.embed_text(text)
        
        return {
            "success": result.get("success", False),
            "embedding": result.get("embedding", []),
            "dimension": result.get("dimension", 0),
            "text_length": len(text),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"Text embedding failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "embedding": [],
            "dimension": 0
        }


def detect_text_language(text: str) -> Dict[str, Any]:
    """
    تشخیص زبان متن
    
    Args:
        text: متن ورودی
    
    Returns:
        dict با زبان و اطمینان
    
    Example:
        >>> result = detect_text_language("Hello world")
        >>> print(f"Language: {result['language']}")
    """
    try:
        from agents.sensor_integration_system import TextSensorWrapper
        
        text_sensor = TextSensorWrapper()
        result = text_sensor.detect_language(text)
        
        return {
            "success": result.get("success", False),
            "language": result.get("language", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "text_length": len(text),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "language": "unknown",
            "confidence": 0.0
        }


# ═══════════════════════════════════════════════════════════════
# MULTI-MODAL SKILLS
# ═══════════════════════════════════════════════════════════════

def process_multimodal_input(input_path: str, auto_detect: bool = True) -> Dict[str, Any]:
    """
    پردازش خودکار ورودی multi-modal
    
    Args:
        input_path: مسیر فایل ورودی
        auto_detect: تشخیص خودکار نوع فایل
    
    Returns:
        dict با نتایج پردازش
    
    Example:
        >>> result = process_multimodal_input("data.jpg")
        >>> print(f"Type: {result['detected_type']}")
    """
    try:
        from agents.sensor_integration_system import SensorIntegrationSystem
        import asyncio
        
        sensor_system = SensorIntegrationSystem()
        
        # Detect input type
        detected_type = sensor_system.detect_input_type(input_path)
        
        # Process with appropriate sensor
        result = asyncio.run(sensor_system.process(input_path))
        
        return {
            "success": result.results.get("success", False),
            "detected_type": detected_type,
            "sensor_type": result.sensor_type,
            "results": result.results,
            "confidence": result.confidence,
            "processing_time": result.processing_time
        }
    except Exception as e:
        logger.error(f"Multi-modal processing failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "detected_type": "unknown"
        }


def analyze_media_file(file_path: str) -> Dict[str, Any]:
    """
    تحلیل جامع فایل media (تصویر، صدا، یا ویدیو)
    
    Args:
        file_path: مسیر فایل
    
    Returns:
        dict با تحلیل کامل
    
    Example:
        >>> result = analyze_media_file("video.mp4")
        >>> print(f"Analysis: {result['analysis']}")
    """
    try:
        from agents.sensor_integration_system import SensorIntegrationSystem
        import asyncio
        
        sensor_system = SensorIntegrationSystem()
        path = Path(file_path)
        
        # Detect type
        file_type = sensor_system.detect_input_type(file_path)
        
        # Get available operations
        operations = sensor_system.get_available_operations(file_type)
        
        # Run all operations
        results = {}
        for operation in operations:
            try:
                result = asyncio.run(sensor_system.process(
                    file_path,
                    sensor_type=file_type,
                    operation=operation
                ))
                results[operation] = result.results
            except Exception as e:
                results[operation] = {"error": str(e)}
        
        return {
            "success": True,
            "file_path": file_path,
            "file_type": file_type,
            "file_size": path.stat().st_size if path.exists() else 0,
            "operations_run": len(operations),
            "results": results
        }
    except Exception as e:
        logger.error(f"Media analysis failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


# ═══════════════════════════════════════════════════════════════
# SKILL REGISTRY
# ═══════════════════════════════════════════════════════════════

SENSOR_SKILLS = {
    # Vision Skills
    "detect_objects_in_image": {
        "function": detect_objects_in_image,
        "description": "تشخیص اشیاء در تصویر با YOLO",
        "category": "vision",
        "parameters": ["image_path", "conf_threshold"],
        "returns": "dict با لیست اشیاء"
    },
    "classify_image_scene": {
        "function": classify_image_scene,
        "description": "طبقه‌بندی صحنه تصویر",
        "category": "vision",
        "parameters": ["image_path"],
        "returns": "dict با نوع صحنه"
    },
    
    # Audio Skills
    "transcribe_audio_file": {
        "function": transcribe_audio_file,
        "description": "تبدیل گفتار به متن با Whisper",
        "category": "audio",
        "parameters": ["audio_path", "language"],
        "returns": "dict با متن"
    },
    "detect_audio_events": {
        "function": detect_audio_events,
        "description": "تشخیص رویدادهای صوتی",
        "category": "audio",
        "parameters": ["audio_path"],
        "returns": "dict با لیست رویدادها"
    },
    
    # Text Skills
    "embed_text_with_bert": {
        "function": embed_text_with_bert,
        "description": "تولید embedding برای متن با BERT",
        "category": "text",
        "parameters": ["text"],
        "returns": "dict با embedding vector"
    },
    "detect_text_language": {
        "function": detect_text_language,
        "description": "تشخیص زبان متن",
        "category": "text",
        "parameters": ["text"],
        "returns": "dict با زبان"
    },
    
    # Multi-modal Skills
    "process_multimodal_input": {
        "function": process_multimodal_input,
        "description": "پردازش خودکار ورودی multi-modal",
        "category": "multimodal",
        "parameters": ["input_path", "auto_detect"],
        "returns": "dict با نتایج"
    },
    "analyze_media_file": {
        "function": analyze_media_file,
        "description": "تحلیل جامع فایل media",
        "category": "multimodal",
        "parameters": ["file_path"],
        "returns": "dict با تحلیل کامل"
    },
}


def get_sensor_skills() -> Dict[str, Dict[str, Any]]:
    """دریافت تمام sensor skills"""
    return SENSOR_SKILLS


def get_sensor_skills_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """دریافت sensor skills بر اساس دسته"""
    return {
        name: skill
        for name, skill in SENSOR_SKILLS.items()
        if skill["category"] == category
    }


# ═══════════════════════════════════════════════════════════════
# AUTO-REGISTRATION
# ═══════════════════════════════════════════════════════════════

def register_sensor_skills():
    """ثبت خودکار sensor skills در skill registry"""
    try:
        from agents.skill_executor import get_global_registry

        registry = get_global_registry()

        for skill_name, skill_info in SENSOR_SKILLS.items():
            registry.register_skill(
                name=skill_name,
                function=skill_info["function"],
                description=skill_info["description"],
                category=skill_info["category"],
                parameters=skill_info.get("parameters"),
            )
        
        logger.info(f"Registered {len(SENSOR_SKILLS)} sensor skills")
        return True
    except Exception as e:
        logger.error(f"Failed to register sensor skills: {e}")
        return False


# Auto-register on import
try:
    register_sensor_skills()
except:
    pass  # Silent fail if registry not available


if __name__ == "__main__":
    # List all sensor skills
    print("Sensor Skills:")
    print("="*60)
    
    for category in ["vision", "audio", "text", "multimodal"]:
        skills = get_sensor_skills_by_category(category)
        print(f"\n{category.upper()} ({len(skills)} skills):")
        for name, info in skills.items():
            print(f"  - {name}: {info['description']}")
    
    print(f"\nTotal: {len(SENSOR_SKILLS)} skills")

