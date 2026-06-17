"""
Wrapper برای استفاده آسان از Model Manager در agents
"""
import logging
from typing import Optional, Any
from contextlib import contextmanager

from core.model_manager import get_model_manager
from core.memory_monitor import get_memory_monitor

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wrapper برای دسترسی آسان به مدل‌ها با مدیریت خودکار حافظه
    """
    
    def __init__(self, model_name: str, auto_cleanup: bool = True):
        """
        Args:
            model_name: نام مدل ثبت شده در ModelManager
            auto_cleanup: پاکسازی خودکار حافظه پس از استفاده
        """
        self.model_name = model_name
        self.auto_cleanup = auto_cleanup
        self._model_manager = get_model_manager()
        self._memory_monitor = get_memory_monitor()
        self._model_cache: Optional[Any] = None
    
    @property
    def model(self) -> Any:
        """دریافت مدل (با lazy loading)"""
        if self._model_cache is None:
            logger.info(f"🔄 Loading model: {self.model_name}")
            self._model_cache = self._model_manager.get_model(self.model_name)
        return self._model_cache
    
    def unload(self) -> None:
        """آزادسازی مدل از حافظه"""
        if self._model_cache is not None:
            logger.info(f"🗑️ Unloading model: {self.model_name}")
            self._model_manager.unload_model(self.model_name)
            self._model_cache = None
    
    @contextmanager
    def use_model(self):
        """
        Context manager برای استفاده موقت از مدل
        
        Usage:
            wrapper = ModelWrapper("embedding")
            with wrapper.use_model() as model:
                result = model.embed_single("test")
        """
        try:
            # بررسی حافظه قبل از استفاده
            memory_status = self._memory_monitor.check_memory()
            if not memory_status["ok"]:
                logger.warning("⚠️ Memory pressure detected, cleaning up...")
                self._memory_monitor.force_cleanup()
                self._model_manager.auto_cleanup()
            
            yield self.model
            
        finally:
            if self.auto_cleanup:
                # پاکسازی حافظه پس از استفاده
                self._memory_monitor.force_cleanup()
    
    def __enter__(self):
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.auto_cleanup:
            self.unload()


def get_embedding_model():
    """دریافت مدل embedding"""
    return ModelWrapper("embedding", auto_cleanup=False)


def get_llm_model():
    """دریافت مدل LLM اصلی"""
    return ModelWrapper("llm_main", auto_cleanup=True)


def get_answer_model():
    """دریافت مدل پاسخ‌دهی"""
    return ModelWrapper("granite_answer", auto_cleanup=True)

