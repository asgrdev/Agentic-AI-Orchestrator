"""
Sensor Integration System
=========================

سیستم یکپارچه برای کار با سنسورها (Vision, Audio, Text) با استفاده از:
- phi4 prompt understanding برای تشخیص نوع ورودی
- Skill system برای اجرای عملیات
- Adaptive orchestrator برای مدیریت فلو

Sensors:
1. Vision: YOLO object detection, scene classification, tracking
2. Audio: Whisper ASR, audio event detection
3. Text: BERT embeddings, language detection, NER

Integration with:
- phi4.understand_and_plan: تشخیص نوع ورودی و plan generation
- SkillExecutor: اجرای skills مربوط به سنسورها
- AdaptiveOrchestrator: مدیریت فلوی پردازش
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SENSOR TYPES & CONFIGS
# ═══════════════════════════════════════════════════════════════

@dataclass
class SensorInput:
    """ورودی سنسور"""
    type: str  # "vision", "audio", "text"
    data: Any  # path to file, raw data, or text
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SensorOutput:
    """خروجی سنسور"""
    sensor_type: str
    results: Dict[str, Any]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ═══════════════════════════════════════════════════════════════
# VISION SENSOR WRAPPER
# ═══════════════════════════════════════════════════════════════

class VisionSensorWrapper:
    """Wrapper برای Vision sensor (YOLO)"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model_path = model_path
        self._model = None
    
    def _load_model(self):
        """بارگذاری lazy model"""
        if self._model is None:
            try:
                from sensors.vision_module.processors.yolo_wrappers import YOLO
                if YOLO is None:
                    raise ImportError("YOLO not available")
                self._model = YOLO(self.model_path)
                logger.info(f"YOLO model loaded: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                raise
    
    def detect_objects(self, image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """تشخیص اشیاء در تصویر"""
        import time
        start_time = time.time()
        
        try:
            self._load_model()
            
            results = self._model(image_path, conf=conf_threshold)
            
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    detections.append({
                        "class": int(box.cls[0]),
                        "class_name": r.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    })
            
            return {
                "success": True,
                "detections": detections,
                "count": len(detections),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def classify_scene(self, image_path: str) -> Dict[str, Any]:
        """طبقه‌بندی صحنه"""
        import time
        start_time = time.time()
        
        try:
            # استفاده از scene classifier
            from sensors.vision_module.processors.scene_classifier import SceneClassifier
            
            classifier = SceneClassifier()
            result = classifier.classify(image_path)
            
            return {
                "success": True,
                "scene": result.get("scene", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Scene classification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }


# ═══════════════════════════════════════════════════════════════
# AUDIO SENSOR WRAPPER
# ═══════════════════════════════════════════════════════════════

class AudioSensorWrapper:
    """Wrapper برای Audio sensor (Whisper)"""
    
    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._processor = None
    
    def _load_processor(self):
        """بارگذاری lazy processor"""
        if self._processor is None:
            try:
                from sensors.audio_module.processors.whisper_asr import WhisperProcessor, WhisperConfig
                
                config = WhisperConfig(model_name=self.model_name)
                self._processor = WhisperProcessor(config)
                logger.info(f"Whisper processor loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                raise
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """تبدیل گفتار به متن"""
        import time
        start_time = time.time()
        
        try:
            self._load_processor()
            
            result = self._processor.transcribe(audio_path, language=language)
            
            return {
                "success": True,
                "text": result.get("text", ""),
                "language": result.get("language", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def detect_audio_events(self, audio_path: str) -> Dict[str, Any]:
        """تشخیص رویدادهای صوتی"""
        import time
        start_time = time.time()
        
        try:
            # استفاده از audio event detector
            from sensors.audio_module.processors.efficientat import EfficientATProcessor
            
            processor = EfficientATProcessor()
            result = processor.detect_events(audio_path)
            
            return {
                "success": True,
                "events": result.get("events", []),
                "count": len(result.get("events", [])),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Audio event detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }


# ═══════════════════════════════════════════════════════════════
# TEXT SENSOR WRAPPER
# ═══════════════════════════════════════════════════════════════

class TextSensorWrapper:
    """Wrapper برای Text sensor (BERT)"""
    
    def __init__(self, model_name: str = "bert-base-cased"):
        self.model_name = model_name
        self._embedder = None
    
    def _load_embedder(self):
        """بارگذاری lazy embedder"""
        if self._embedder is None:
            try:
                from sensors.text_module.processors.bert_embedder import BertEmbedder, BertEmbedderConfig
                
                config = BertEmbedderConfig(model_name_or_path=self.model_name)
                self._embedder = BertEmbedder(config)
                logger.info(f"BERT embedder loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load BERT: {e}")
                raise
    
    def embed_text(self, text: str) -> Dict[str, Any]:
        """تولید embedding برای متن"""
        import time
        start_time = time.time()
        
        try:
            self._load_embedder()
            
            embedding = self._embedder.embed(text)
            
            return {
                "success": True,
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "dimension": len(embedding),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """تشخیص زبان متن"""
        import time
        start_time = time.time()
        
        try:
            from sensors.text_module.processors.language import LanguageDetector
            
            detector = LanguageDetector()
            result = detector.detect(text)
            
            return {
                "success": True,
                "language": result.get("language", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }


# ═══════════════════════════════════════════════════════════════
# SENSOR INTEGRATION SYSTEM
# ═══════════════════════════════════════════════════════════════

class SensorIntegrationSystem:
    """
    سیستم یکپارچه برای کار با سنسورها
    
    Features:
    - Auto-detection: تشخیص خودکار نوع ورودی
    - Multi-modal: پشتیبانی از Vision, Audio, Text
    - Skill integration: یکپارچگی با skill system
    - phi4 integration: استفاده از phi4 برای planning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize sensors (lazy loading)
        self.vision = VisionSensorWrapper(
            model_path=self.config.get("vision_model", "yolov8n.pt")
        )
        self.audio = AudioSensorWrapper(
            model_name=self.config.get("audio_model", "base")
        )
        self.text = TextSensorWrapper(
            model_name=self.config.get("text_model", "bert-base-cased")
        )
        
        logger.info("SensorIntegrationSystem initialized")
    
    def detect_input_type(self, input_data: Union[str, Path, bytes]) -> str:
        """تشخیص نوع ورودی"""
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            
            # Check file extension
            ext = path.suffix.lower()
            
            # Image extensions
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']:
                return "vision"
            
            # Audio extensions
            elif ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']:
                return "audio"
            
            # Video extensions (treat as vision)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                return "vision"
            
            # Text file or raw text
            elif ext in ['.txt', '.md', '.json', '.csv'] or not path.exists():
                return "text"
        
        # Default to text for string input
        elif isinstance(input_data, str):
            return "text"
        
        # Binary data - try to detect
        elif isinstance(input_data, bytes):
            # Simple heuristic: check first bytes
            if input_data[:4] == b'\xff\xd8\xff':  # JPEG
                return "vision"
            elif input_data[:4] == b'RIFF':  # WAV
                return "audio"
        
        return "text"  # Default
    
    async def process(self, input_data: Union[str, Path, bytes], 
                     sensor_type: Optional[str] = None,
                     operation: Optional[str] = None) -> SensorOutput:
        """
        پردازش ورودی با سنسور مناسب
        
        Args:
            input_data: داده ورودی (path, text, or bytes)
            sensor_type: نوع سنسور (auto-detect if None)
            operation: عملیات مورد نظر (auto-select if None)
        
        Returns:
            SensorOutput با نتایج
        """
        import time
        start_time = time.time()
        
        # Auto-detect sensor type
        if sensor_type is None:
            sensor_type = self.detect_input_type(input_data)
        
        logger.info(f"Processing with {sensor_type} sensor")
        
        try:
            if sensor_type == "vision":
                result = await self._process_vision(input_data, operation)
            elif sensor_type == "audio":
                result = await self._process_audio(input_data, operation)
            elif sensor_type == "text":
                result = await self._process_text(input_data, operation)
            else:
                raise ValueError(f"Unknown sensor type: {sensor_type}")
            
            return SensorOutput(
                sensor_type=sensor_type,
                results=result,
                confidence=result.get("confidence", 0.0),
                processing_time=time.time() - start_time
            )
        
        except Exception as e:
            logger.error(f"Sensor processing failed: {e}")
            return SensorOutput(
                sensor_type=sensor_type,
                results={"success": False, "error": str(e)},
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    async def _process_vision(self, input_data: Union[str, Path], 
                             operation: Optional[str]) -> Dict[str, Any]:
        """پردازش با vision sensor"""
        if operation == "classify":
            return self.vision.classify_scene(str(input_data))
        else:  # default: detect
            return self.vision.detect_objects(str(input_data))
    
    async def _process_audio(self, input_data: Union[str, Path], 
                            operation: Optional[str]) -> Dict[str, Any]:
        """پردازش با audio sensor"""
        if operation == "detect_events":
            return self.audio.detect_audio_events(str(input_data))
        else:  # default: transcribe
            return self.audio.transcribe(str(input_data))
    
    async def _process_text(self, input_data: str, 
                           operation: Optional[str]) -> Dict[str, Any]:
        """پردازش با text sensor"""
        if operation == "detect_language":
            return self.text.detect_language(input_data)
        else:  # default: embed
            return self.text.embed_text(input_data)
    
    def get_available_operations(self, sensor_type: str) -> List[str]:
        """لیست عملیات‌های موجود برای هر سنسور"""
        operations = {
            "vision": ["detect_objects", "classify_scene", "track_objects"],
            "audio": ["transcribe", "detect_events"],
            "text": ["embed_text", "detect_language", "extract_entities"]
        }
        return operations.get(sensor_type, [])


# ═══════════════════════════════════════════════════════════════
# PHI4 INTEGRATION
# ═══════════════════════════════════════════════════════════════

class SensorPhi4Integration:
    """
    یکپارچگی سنسورها با phi4 prompt understanding
    
    phi4 می‌تواند:
    1. نوع ورودی را تشخیص دهد
    2. عملیات مناسب را انتخاب کند
    3. Plan برای پردازش multi-modal تولید کند
    """
    
    def __init__(self, sensor_system: SensorIntegrationSystem, phi4_client):
        self.sensor_system = sensor_system
        self.phi4 = phi4_client
    
    async def understand_and_process(self, query: str, input_data: Any) -> Dict[str, Any]:
        """
        استفاده از phi4 برای فهم query و پردازش با سنسور مناسب
        
        Example queries:
        - "What objects are in this image?"
        - "Transcribe this audio file"
        - "Detect the language of this text"
        """
        # Use phi4 to understand query
        plan = await self.phi4.understand_and_plan(query)
        
        # Extract sensor requirements from plan
        sensor_type = self._extract_sensor_type(plan)
        operation = self._extract_operation(plan)
        
        # Process with appropriate sensor
        result = await self.sensor_system.process(
            input_data,
            sensor_type=sensor_type,
            operation=operation
        )
        
        return {
            "query": query,
            "plan": plan,
            "sensor_type": sensor_type,
            "operation": operation,
            "result": result
        }
    
    def _extract_sensor_type(self, plan: Dict[str, Any]) -> Optional[str]:
        """استخراج نوع سنسور از plan"""
        # Check for sensor-related keywords in plan
        plan_text = str(plan).lower()
        
        if any(word in plan_text for word in ["image", "photo", "picture", "video", "visual"]):
            return "vision"
        elif any(word in plan_text for word in ["audio", "sound", "speech", "voice", "transcribe"]):
            return "audio"
        elif any(word in plan_text for word in ["text", "language", "embed", "nlp"]):
            return "text"
        
        return None
    
    def _extract_operation(self, plan: Dict[str, Any]) -> Optional[str]:
        """استخراج عملیات از plan"""
        plan_text = str(plan).lower()
        
        # Vision operations
        if "detect" in plan_text or "object" in plan_text:
            return "detect"
        elif "classify" in plan_text or "scene" in plan_text:
            return "classify"
        
        # Audio operations
        elif "transcribe" in plan_text or "speech" in plan_text:
            return "transcribe"
        elif "event" in plan_text or "sound" in plan_text:
            return "detect_events"
        
        # Text operations
        elif "language" in plan_text:
            return "detect_language"
        elif "embed" in plan_text:
            return "embed"
        
        return None


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize system
    sensor_system = SensorIntegrationSystem()
    
    # Example 1: Auto-detect and process
    print("Example 1: Auto-detect")
    print("="*60)
    
    # Detect input type
    input_type = sensor_system.detect_input_type("image.jpg")
    print(f"Detected type: {input_type}")
    
    # Get available operations
    operations = sensor_system.get_available_operations(input_type)
    print(f"Available operations: {operations}")
    
    print("\n✅ Sensor Integration System ready!")

