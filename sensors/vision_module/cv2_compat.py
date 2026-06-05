from __future__ import annotations

import os
from typing import Any, Optional, Tuple


def _env_true(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _numpy_major() -> int:
    try:
        import numpy as np

        return int(str(getattr(np, "__version__", "0")).split(".", 1)[0])
    except Exception:
        return 0


def _load_cv2() -> Tuple[Optional[Any], str]:
    # Strict by default: do not silently bypass core vision dependency.
    strict_modules = _env_true("MAIN_APP_STRICT_MODULES", default=True)
    force_import = _env_true("MAIN_APP_FORCE_CV2_IMPORT", default=False)
    allow_bypass = _env_true("MAIN_APP_ALLOW_MODULE_BYPASS", default=False)
    np_major = _numpy_major()
    if (not force_import) and np_major >= 2:
        msg = (
            f"cv2_numpy_abi_incompatible:numpy_major={np_major}. "
            "Install numpy<2 and reinstall opencv/ultralytics wheels."
        )
        if strict_modules and (not allow_bypass):
            raise RuntimeError(msg)
        return None, msg
    try:
        import cv2  # type: ignore

        return cv2, ""
    except Exception as e:
        if strict_modules and (not allow_bypass):
            raise
        return None, repr(e)


cv2, CV2_IMPORT_ERROR = _load_cv2()
CV2_AVAILABLE = cv2 is not None
