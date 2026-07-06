"""
تعیین مسیر مدل‌ها — اول لوکالِ پروژه، بعد مسیر fallback.

قاعده: هر مدل اول از `<ریشه پروژه>/models/<model_name>` (یا مسیر متغیر
محیطی MODELS_DIR) جستجو می‌شود؛ اگر آنجا نبود، مسیر fallback (مسیرهای
مطلق قدیمی) استفاده می‌شود. این‌طوری کافی است مدل را داخل پوشه‌ی
`models/` پروژه بگذارید تا بدون تغییر کد لود شود.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_logged: set[str] = set()


def resolve_model_path(model_name: str, fallback: Optional[str] = None) -> str:
    """
    مسیر نهایی مدل:
      1. $MODELS_DIR/<model_name>          (اگر MODELS_DIR ست شده باشد)
      2. <project>/models/<model_name>
      3. fallback (مسیر مطلق قدیمی)
    """
    candidates = []
    env_dir = os.getenv("MODELS_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / model_name)
    candidates.append(PROJECT_ROOT / "models" / model_name)

    for cand in candidates:
        if cand.exists():
            resolved = str(cand)
            if model_name not in _logged:
                _logged.add(model_name)
                logger.info("📁 Model '%s' → local: %s", model_name, resolved)
            return resolved

    result = fallback or str(PROJECT_ROOT / "models" / model_name)
    if model_name not in _logged:
        _logged.add(model_name)
        logger.info(
            "📁 Model '%s' not found in %s — using fallback: %s",
            model_name, [str(c.parent) for c in candidates], result,
        )
    return result
