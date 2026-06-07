"""
ابزارهای نظارت و مدیریت حافظه
"""
import gc
import logging
import psutil
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """اسنپ‌شات از وضعیت حافظه"""
    process_rss_mb: float  # Resident Set Size
    process_vms_mb: float  # Virtual Memory Size
    process_percent: float
    system_available_mb: float
    system_used_mb: float
    system_percent: float
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    gpu_percent: Optional[float] = None


class MemoryMonitor:
    """
    نظارت بر مصرف حافظه CPU و GPU
    """
    
    def __init__(self, warning_threshold: float = 80.0):
        """
        Args:
            warning_threshold: درصد حافظه که در آن هشدار صادر شود
        """
        self.warning_threshold = warning_threshold
        self.process = psutil.Process(os.getpid())
        
    def get_snapshot(self) -> MemorySnapshot:
        """دریافت اسنپ‌شات فعلی حافظه"""
        # CPU Memory
        mem_info = self.process.memory_info()
        system_mem = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            process_rss_mb=mem_info.rss / 1024**2,
            process_vms_mb=mem_info.vms / 1024**2,
            process_percent=self.process.memory_percent(),
            system_available_mb=system_mem.available / 1024**2,
            system_used_mb=system_mem.used / 1024**2,
            system_percent=system_mem.percent,
        )
        
        # GPU Memory (if available)
        try:
            import torch
            if torch.cuda.is_available():
                snapshot.gpu_allocated_mb = torch.cuda.memory_allocated() / 1024**2
                snapshot.gpu_reserved_mb = torch.cuda.memory_reserved() / 1024**2
                total_memory = torch.cuda.get_device_properties(0).total_memory
                snapshot.gpu_percent = (snapshot.gpu_allocated_mb / (total_memory / 1024**2)) * 100
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't provide detailed memory stats
                snapshot.gpu_allocated_mb = 0.0
                snapshot.gpu_reserved_mb = 0.0
        except ImportError:
            pass
        
        return snapshot
    
    def check_memory(self) -> Dict[str, Any]:
        """بررسی وضعیت حافظه و صدور هشدار در صورت نیاز"""
        snapshot = self.get_snapshot()
        
        status = {
            "ok": True,
            "warnings": [],
            "snapshot": snapshot,
        }
        
        # بررسی حافظه سیستم
        if snapshot.system_percent > self.warning_threshold:
            status["ok"] = False
            status["warnings"].append(
                f"⚠️ System memory high: {snapshot.system_percent:.1f}%"
            )
        
        # بررسی حافظه GPU
        if snapshot.gpu_percent and snapshot.gpu_percent > self.warning_threshold:
            status["ok"] = False
            status["warnings"].append(
                f"⚠️ GPU memory high: {snapshot.gpu_percent:.1f}%"
            )
        
        # لاگ هشدارها
        for warning in status["warnings"]:
            logger.warning(warning)
        
        return status
    
    def force_cleanup(self) -> None:
        """پاکسازی اجباری حافظه"""
        logger.info("🧹 Starting memory cleanup...")
        
        # Python garbage collection
        collected = gc.collect()
        logger.info(f"   Collected {collected} objects")
        
        # PyTorch cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("   CUDA cache cleared")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("   MPS cache cleared")
        except ImportError:
            pass
        
        # MLX cleanup
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
            logger.info("   MLX cache cleared")
        except (ImportError, AttributeError):
            pass
        
        logger.info("✅ Memory cleanup completed")
    
    def log_stats(self) -> None:
        """لاگ آمار حافظه"""
        snapshot = self.get_snapshot()
        
        logger.info("📊 Memory Statistics:")
        logger.info(f"   Process RSS: {snapshot.process_rss_mb:.1f} MB")
        logger.info(f"   Process %: {snapshot.process_percent:.1f}%")
        logger.info(f"   System Available: {snapshot.system_available_mb:.1f} MB")
        logger.info(f"   System Used: {snapshot.system_used_mb:.1f} MB ({snapshot.system_percent:.1f}%)")
        
        if snapshot.gpu_allocated_mb is not None:
            logger.info(f"   GPU Allocated: {snapshot.gpu_allocated_mb:.1f} MB")
            logger.info(f"   GPU Reserved: {snapshot.gpu_reserved_mb:.1f} MB")
            if snapshot.gpu_percent:
                logger.info(f"   GPU %: {snapshot.gpu_percent:.1f}%")
    
    @contextmanager
    def track_memory(self, operation_name: str = "Operation"):
        """
        Context manager برای ردیابی مصرف حافظه یک عملیات
        
        Usage:
            with monitor.track_memory("Model Loading"):
                model = load_model()
        """
        logger.info(f"🔍 Tracking memory for: {operation_name}")
        
        before = self.get_snapshot()
        
        try:
            yield
        finally:
            after = self.get_snapshot()
            
            delta_rss = after.process_rss_mb - before.process_rss_mb
            delta_gpu = 0.0
            if after.gpu_allocated_mb and before.gpu_allocated_mb:
                delta_gpu = after.gpu_allocated_mb - before.gpu_allocated_mb
            
            logger.info(f"📈 {operation_name} Memory Delta:")
            logger.info(f"   RSS: {delta_rss:+.1f} MB")
            if delta_gpu != 0.0:
                logger.info(f"   GPU: {delta_gpu:+.1f} MB")


# Global instance
_memory_monitor = None


def get_memory_monitor() -> MemoryMonitor:
    """دریافت نمونه سینگلتون MemoryMonitor"""
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor

# Made with Bob
