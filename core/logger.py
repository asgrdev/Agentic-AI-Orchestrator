"""
تنظیم متمرکز logging برای کل سیستم.

قبلاً basicConfig داخل ماژول‌های کتابخانه‌ای (embedding_generator) صدا زده
می‌شد؛ نتیجه: فرمت‌های ناهماهنگ و لاگ‌های تکراری. حالا فقط entrypoint ها
(main*.py) این تابع را صدا می‌زنند.
"""
from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# کتابخانه‌هایی که در سطح INFO بیش از حد شلوغ‌اند
# (خطوط «HTTP Request: GET ...» در لاگ از httpx می‌آمد)
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "weaviate",
    "gradio",
    "huggingface_hub",
    "filelock",
    "PIL",
)


def setup_logging(
    level: int | str | None = None,
    quiet_noisy_libs: bool = True,
    fmt: str = _DEFAULT_FORMAT,
) -> None:
    """
    راه‌اندازی root logger — فقط از entrypoint صدا زده شود.

    سطح لاگ با LOG_LEVEL از environment قابل override است
    (مثلاً LOG_LEVEL=DEBUG python main_adaptive.py).
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()

    # idempotent: اگر handler ما قبلاً نصب شده، فقط سطح را به‌روز کن —
    # نصب دوباره یعنی هر پیام دوبار چاپ می‌شود
    for h in root.handlers:
        if getattr(h, "_agentic_rag_handler", False):
            root.setLevel(level)
            return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    handler._agentic_rag_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)

    if quiet_noisy_libs:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
