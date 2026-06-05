# external_sources/data_collector/timeout.py

import functools
import time
import socket
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from loguru import logger
from external_sources.data_collector.models import CollectionResult, DataSource


def with_timeout(timeout_sec: float):
    """
    Decorator برای اعمال timeout روی متدهای collector.
    ترکیبی از ThreadPoolExecutor و socket timeout برای پوشش همه حالت‌ها.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            
            # ذخیره timeout قبلی socket
            original_timeout = socket.getdefaulttimeout()
            
            # تنظیم timeout برای network operations (کمی کمتر از timeout کلی)
            socket.setdefaulttimeout(max(1.0, timeout_sec - 0.5))
            
            try:
                # استفاده از ThreadPoolExecutor برای timeout کلی
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    
                    try:
                        result = future.result(timeout=timeout_sec)
                        return result
                    
                    except FuturesTimeoutError:
                        # timeout رخ داد
                        elapsed = time.time() - start
                        logger.warning(
                            f"⏱ [{func.__qualname__}] timeout after {elapsed:.1f}s "
                            f"(limit: {timeout_sec}s)"
                        )
                        
                        # تشخیص source
                        source = _get_source_from_args(args, func.__name__)
                        
                        return CollectionResult(
                            success=False,
                            source=source,
                            total=0,
                            items=[],
                            errors=[f"Timeout after {timeout_sec}s in {func.__name__}"],
                            elapsed_sec=elapsed,
                        )
                    
                    except Exception as e:
                        # خطای دیگر در حین اجرا
                        elapsed = time.time() - start
                        logger.error(f"❌ [{func.__qualname__}] error: {e}")
                        
                        source = _get_source_from_args(args, func.__name__)
                        
                        return CollectionResult(
                            success=False,
                            source=source,
                            total=0,
                            items=[],
                            errors=[f"Error in {func.__name__}: {str(e)}"],
                            elapsed_sec=elapsed,
                        )
            
            finally:
                # بازگردانی timeout قبلی socket
                socket.setdefaulttimeout(original_timeout)
        
        return wrapper
    return decorator


def _get_source_from_args(args, func_name: str) -> DataSource:
    """
    تشخیص source از args یا نام تابع
    """
    # اگر args[0] یک instance از collector است
    if args and hasattr(args[0], "_default_source"):
        source = getattr(args[0], "_default_source", None)
        if source is not None:
            return source
    
    # تشخیص از نام تابع
    return _infer_source_from_name(func_name)


def _infer_source_from_name(func_name: str) -> DataSource:
    """
    تشخیص source از نام تابع
    """
    name = func_name.lower()
    
    if "stock" in name or "yfinance" in name or "ticker" in name:
        return DataSource.YFINANCE
    
    if "fred" in name or "economic" in name:
        return DataSource.FRED
    
    if "rss" in name or "feed" in name:
        return DataSource.RSS_FEED
    
    if "telegram" in name:
        return DataSource.TELEGRAM
    
    if "reddit" in name:
        return DataSource.REDDIT
    
    if "twitter" in name or "tweet" in name:
        return DataSource.TWITTER
    
    # پیش‌فرض
    return DataSource.WEB_SCRAPE
