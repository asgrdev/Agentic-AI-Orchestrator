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
import warnings

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# هشدارهای deprecation داخل کتابخانه‌های ثالث که از این پروژه قابل رفع
# نیستند و فقط لاگ را شلوغ می‌کنند — با message هدفمند فیلتر می‌شوند تا
# هشدارهای واقعی کد خودمان مخفی نشوند.
_THIRD_PARTY_WARNING_PATTERNS = (
    # gradio/routes.py از ثابت deprecated شده‌ی starlette استفاده می‌کند
    r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated",
)


def _suppress_third_party_warnings() -> None:
    for pattern in _THIRD_PARTY_WARNING_PATTERNS:
        warnings.filterwarnings(
            "ignore", message=pattern, category=DeprecationWarning
        )
    _patch_deprecated_status_constants()


def _patch_deprecated_status_constants() -> None:
    """
    خفه‌کردن هشدار 422 گرادیو در «منبع» — مستقل از فیلترهای warnings.

    gradio/routes.py در هر request ثابت HTTP_422_UNPROCESSABLE_ENTITY را
    می‌خواند؛ starlette جدید این نام را حذف کرده و از طریق __getattr__ ماژول
    هر بار DeprecationWarning می‌دهد. فیلتر warnings ترتیبی است و اگر
    کتابخانه‌ای بعداً فیلتر خودش را جلوی لیست بگذارد دوباره چاپ می‌شود؛
    ولی اگر نام را یک‌بار در dict ماژول بگذاریم، __getattr__ دیگر هرگز
    اجرا نمی‌شود — قطعی و بدون اثر جانبی (همان عدد 422 است).
    """
    import importlib

    for mod_name in ("starlette.status", "fastapi.status"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        try:
            if "HTTP_422_UNPROCESSABLE_ENTITY" not in vars(mod):
                value = vars(mod).get("HTTP_422_UNPROCESSABLE_CONTENT", 422)
                setattr(mod, "HTTP_422_UNPROCESSABLE_ENTITY", value)
        except Exception:
            pass

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

    _suppress_third_party_warnings()

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
