"""
مدیریت متمرکز مدل‌ها با قابلیت lazy loading و memory management
"""
import gc
import logging
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
from pathlib import Path
import threading
import weakref

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """تنظیمات یک مدل"""
    model_path: str
    model_type: Literal["embedding", "llm", "reasoning"]
    device: Optional[str] = None
    use_fp16: bool = True
    use_4bit: bool = False
    max_memory_mb: Optional[int] = None
    auto_unload: bool = True  # آزادسازی خودکار پس از استفاده
    cache_size: int = 1000  # برای embedding models


class ModelManager:
    """
    مدیر متمرکز برای بارگذاری و آزادسازی مدل‌ها
    
    ویژگی‌ها:
    - Lazy loading: مدل‌ها فقط در صورت نیاز بارگذاری می‌شوند
    - Auto unloading: آزادسازی خودکار مدل‌های استفاده نشده
    - Memory monitoring: نظارت بر مصرف حافظه
    - Thread-safe: امن برای استفاده همزمان
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._models: Dict[str, Any] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._usage_count: Dict[str, int] = {}
        self._lock = threading.Lock()
        
        logger.info("✅ ModelManager initialized")
    
    def register_model(self, name: str, config: ModelConfig) -> None:
        """ثبت تنظیمات یک مدل بدون بارگذاری آن"""
        with self._lock:
            self._configs[name] = config
            self._usage_count[name] = 0
            logger.info(f"📝 Model registered: {name} ({config.model_type})")
    
    def get_model(self, name: str) -> Any:
        """
        دریافت مدل (با lazy loading)
        
        Args:
            name: نام مدل ثبت شده
            
        Returns:
            نمونه مدل بارگذاری شده
        """
        with self._lock:
            if name not in self._configs:
                raise ValueError(f"Model '{name}' not registered")
            
            # اگر مدل بارگذاری نشده، آن را بارگذاری کن
            # (لاگ سطح debug — لود واقعی وزن‌ها را ModelGate با جزئیات حافظه لاگ می‌کند)
            if name not in self._models:
                logger.debug(f"🔄 Loading model client: {name}")
                self._models[name] = self._load_model(name)
            
            self._usage_count[name] += 1
            return self._models[name]
    
    def _load_model(self, name: str) -> Any:
        """بارگذاری واقعی مدل بر اساس نوع آن"""
        config = self._configs[name]
        
        try:
            if config.model_type == "embedding":
                return self._load_embedding_model(config)
            elif config.model_type == "llm":
                return self._load_llm_model(config)
            elif config.model_type == "reasoning":
                return self._load_reasoning_model(config)
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")
        except Exception as e:
            logger.error(f"❌ Failed to load model {name}: {e}")
            raise
    
    def _load_embedding_model(self, config: ModelConfig):
        """بارگذاری مدل embedding"""
        from ingestion.embedding_generator import Qwen3EmbeddingClient
        
        return Qwen3EmbeddingClient(
            model_path=config.model_path,
            device=config.device,
            use_fp16=config.use_fp16,
            use_4bit=config.use_4bit,
            cache_embeddings=True,
            batch_size=16,  # کاهش batch size برای کاهش مصرف حافظه
        )
    
    def _load_llm_model(self, config: ModelConfig):
        """بارگذاری مدل LLM"""
        from llm.mlx_granite import MlxGraniteAnswerGenerator, MlxGraniteConfig
        
        mlx_config = MlxGraniteConfig(
            model_path=config.model_path,
            max_tokens=1024,  # کاهش max_tokens
            temperature=0.7,
        )
        return MlxGraniteAnswerGenerator(mlx_config)
    
    def _load_reasoning_model(self, config: ModelConfig):
        """بارگذاری مدل reasoning"""
        # نام کلاس MlxGraniteClient است و فقط MlxGraniteConfig می‌پذیرد —
        # import/امضای قبلی (MLXGraniteClient, backend=...) اصلاً وجود نداشت
        from llm.granite_client.mlx_client import MlxGraniteClient
        from llm.granite_client.config import MlxGraniteConfig

        return MlxGraniteClient(MlxGraniteConfig(model_path=config.model_path))
    
    def unload_model(self, name: str) -> None:
        """آزادسازی یک مدل از حافظه"""
        with self._lock:
            if name in self._models:
                model = self._models[name]
                
                # فراخوانی متد free_memory اگر وجود دارد
                if hasattr(model, 'free_memory'):
                    model.free_memory()
                
                # حذف مدل
                del self._models[name]
                
                # پاکسازی حافظه
                self._cleanup_memory()
                
                logger.info(f"🗑️ Model unloaded: {name}")
    
    def unload_all(self) -> None:
        """آزادسازی تمام مدل‌ها"""
        with self._lock:
            model_names = list(self._models.keys())
            for name in model_names:
                self.unload_model(name)
            
            logger.info("🗑️ All models unloaded")
    
    def _cleanup_memory(self) -> None:
        """پاکسازی حافظه GPU/CPU"""
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
        
        try:
            import mlx.core as mx
            # MLX memory cleanup — mx.metal.clear_cache در نسخه‌های جدید deprecated است
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()
        except (ImportError, AttributeError):
            pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """دریافت آمار مصرف حافظه"""
        stats = {
            "loaded_models": list(self._models.keys()),
            "registered_models": list(self._configs.keys()),
            "usage_count": self._usage_count.copy(),
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                stats["cuda_memory_allocated"] = torch.cuda.memory_allocated() / 1024**2  # MB
                stats["cuda_memory_reserved"] = torch.cuda.memory_reserved() / 1024**2
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                stats["mps_available"] = True
        except ImportError:
            pass
        
        return stats
    
    def auto_cleanup(self, threshold_mb: int = 4096) -> None:
        """
        پاکسازی خودکار مدل‌های کم استفاده در صورت پر شدن حافظه
        
        Args:
            threshold_mb: حد آستانه حافظه به مگابایت
        """
        stats = self.get_memory_stats()
        
        # بررسی مصرف حافظه
        memory_used = stats.get("cuda_memory_allocated", 0)
        
        if memory_used > threshold_mb:
            logger.warning(f"⚠️ Memory usage high: {memory_used:.0f}MB")
            
            # مرتب‌سازی مدل‌ها بر اساس تعداد استفاده
            sorted_models = sorted(
                self._usage_count.items(),
                key=lambda x: x[1]
            )
            
            # آزادسازی مدل‌های کم استفاده
            for name, count in sorted_models:
                if name in self._models:
                    config = self._configs[name]
                    if config.auto_unload:
                        self.unload_model(name)
                        logger.info(f"🧹 Auto-unloaded: {name} (usage: {count})")
                        break
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_all()
    
    def __repr__(self) -> str:
        return (
            f"ModelManager("
            f"loaded={len(self._models)}, "
            f"registered={len(self._configs)})"
        )


# Global instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """دریافت نمونه سینگلتون ModelManager"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

