"""
تست‌های سیستم مدیریت حافظه
"""
import sys
from pathlib import Path

# اضافه کردن مسیر پروژه به sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def test_model_manager_basic():
    """تست پایه Model Manager"""
    from core.model_manager import get_model_manager, ModelConfig
    
    logger.info("=== Test 1: Model Manager Basic ===")
    
    manager = get_model_manager()
    
    # ثبت یک مدل تست
    manager.register_model(
        "test_embedding",
        ModelConfig(
            model_path="/Users/dbk/Desktop/agentic-graph-RAG/models/Qwen3-Embedding-0.6B",
            model_type="embedding",
            device="mps",
            use_fp16=True,
            auto_unload=True,
        )
    )
    
    logger.info("✅ Model registered successfully")
    
    # بررسی آمار
    stats = manager.get_memory_stats()
    logger.info(f"Stats: {stats}")
    
    assert "test_embedding" in stats["registered_models"]
    logger.info("✅ Test 1 passed")


def test_memory_monitor():
    """تست Memory Monitor"""
    from core.memory_monitor import get_memory_monitor
    
    logger.info("=== Test 2: Memory Monitor ===")
    
    monitor = get_memory_monitor()
    
    # دریافت اسنپ‌شات
    snapshot = monitor.get_snapshot()
    logger.info(f"Process RSS: {snapshot.process_rss_mb:.1f} MB")
    logger.info(f"System Used: {snapshot.system_used_mb:.1f} MB ({snapshot.system_percent:.1f}%)")
    
    # بررسی حافظه
    status = monitor.check_memory()
    logger.info(f"Memory OK: {status['ok']}")
    
    # لاگ آمار
    monitor.log_stats()
    
    logger.info("✅ Test 2 passed")


def test_model_wrapper():
    """تست Model Wrapper"""
    from core.model_wrapper import ModelWrapper
    from core.model_manager import get_model_manager, ModelConfig
    
    logger.info("=== Test 3: Model Wrapper ===")
    
    # ثبت مدل
    manager = get_model_manager()
    if "test_embedding" not in manager._configs:
        manager.register_model(
            "test_embedding",
            ModelConfig(
                model_path="/Users/dbk/Desktop/agentic-graph-RAG/models/Qwen3-Embedding-0.6B",
                model_type="embedding",
                device="mps",
                use_fp16=True,
                auto_unload=False,
            )
        )
    
    # استفاده از wrapper
    wrapper = ModelWrapper("test_embedding", auto_cleanup=False)
    
    logger.info("Getting model...")
    model = wrapper.model
    logger.info(f"Model loaded: {type(model)}")
    
    # تست embedding
    logger.info("Testing embedding...")
    result = model.embed_single("test text")
    logger.info(f"Embedding dim: {len(result['embedding'])}")
    
    assert len(result['embedding']) > 0
    logger.info("✅ Test 3 passed")


def test_model_loading_unloading():
    """تست بارگذاری و آزادسازی مدل"""
    from core.model_manager import get_model_manager
    from core.memory_monitor import get_memory_monitor
    
    logger.info("=== Test 4: Model Loading/Unloading ===")
    
    manager = get_model_manager()
    monitor = get_memory_monitor()
    
    # حافظه قبل از بارگذاری
    before = monitor.get_snapshot()
    logger.info(f"Memory before: {before.process_rss_mb:.1f} MB")
    
    # بارگذاری مدل
    with monitor.track_memory("Model Loading"):
        model = manager.get_model("test_embedding")
        logger.info("Model loaded")
    
    # حافظه بعد از بارگذاری
    after_load = monitor.get_snapshot()
    logger.info(f"Memory after load: {after_load.process_rss_mb:.1f} MB")
    delta = after_load.process_rss_mb - before.process_rss_mb
    logger.info(f"Memory delta: {delta:+.1f} MB")
    
    # آزادسازی مدل
    with monitor.track_memory("Model Unloading"):
        manager.unload_model("test_embedding")
        logger.info("Model unloaded")
    
    # حافظه بعد از آزادسازی
    after_unload = monitor.get_snapshot()
    logger.info(f"Memory after unload: {after_unload.process_rss_mb:.1f} MB")
    
    logger.info("✅ Test 4 passed")


def test_context_manager():
    """تست استفاده از context manager"""
    from core.model_wrapper import ModelWrapper
    from core.memory_monitor import get_memory_monitor
    
    logger.info("=== Test 5: Context Manager ===")
    
    monitor = get_memory_monitor()
    wrapper = ModelWrapper("test_embedding", auto_cleanup=True)
    
    # استفاده با context manager
    with monitor.track_memory("Context Manager Usage"):
        with wrapper.use_model() as model:
            result = model.embed_single("context manager test")
            logger.info(f"Embedding generated: {len(result['embedding'])} dims")
    
    logger.info("✅ Test 5 passed")


def run_all_tests():
    """اجرای تمام تست‌ها"""
    logger.info("=" * 60)
    logger.info("Starting Memory Management Tests")
    logger.info("=" * 60)
    
    try:
        test_model_manager_basic()
        test_memory_monitor()
        test_model_wrapper()
        test_model_loading_unloading()
        test_context_manager()
        
        logger.info("=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_all_tests()

