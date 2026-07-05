"""
ابزار پل زدن sync ↔ async بدون نشت event loop.

الگوی قدیمی `asyncio.get_event_loop().run_until_complete(...)` در context
های sync هر بار یک loop می‌سازد که هیچ‌وقت بسته نمی‌شود — همان
«unclosed event loop» ResourceWarning هایی که در لاگ startup دیده می‌شد.

run_sync() به‌جای آن از یک background loop مشترک (یک thread ثابت) استفاده
می‌کند: از هر جای sync قابل صدا زدن است (حتی از داخل handler هایی که خودشان
در thread ای بدون loop اجرا می‌شوند) و در پایان پروسه تمیز بسته می‌شود.
"""
from __future__ import annotations

import asyncio
import atexit
import threading
from typing import Any, Coroutine, Optional

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_lock = threading.Lock()


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is not None and _loop.is_running():
        return _loop

    with _lock:
        if _loop is not None and _loop.is_running():
            return _loop

        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name="shared-async-loop",
            daemon=True,
        )
        thread.start()
        _loop, _loop_thread = loop, thread
        return loop


def _shutdown() -> None:
    global _loop
    if _loop is not None and _loop.is_running():
        _loop.call_soon_threadsafe(_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=2)
        _loop.close()
        _loop = None


atexit.register(_shutdown)


def run_sync(coro: Coroutine[Any, Any, Any], timeout: Optional[float] = None) -> Any:
    """
    اجرای یک coroutine از کد sync.

    برخلاف asyncio.run:
    - loop جدیدی نمی‌سازد و رها نمی‌کند (loop مشترک، یک بار)
    - از thread ای که خودش event loop در حال اجرا دارد هم قابل استفاده است
      (کار به loop مشترک سپرده می‌شود، نه loop جاری)
    """
    try:
        asyncio.get_running_loop()
        in_async_context = True
    except RuntimeError:
        in_async_context = False

    if in_async_context:
        # داخل loop جاری block نمی‌کنیم — به background loop می‌سپاریم؛
        # توجه: این حالت فقط برای سازگاری است، در کد async مستقیم await کنید
        future = asyncio.run_coroutine_threadsafe(coro, _ensure_background_loop())
        return future.result(timeout)

    future = asyncio.run_coroutine_threadsafe(coro, _ensure_background_loop())
    return future.result(timeout)
