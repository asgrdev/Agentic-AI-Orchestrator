from __future__ import annotations
import re
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def normalize_text(text: str, normalize_whitespace: bool = True, strip_control_chars: bool = True) -> str:
    out = text or ""
    if strip_control_chars:
        out = _CONTROL_CHARS.sub("", out)
    if normalize_whitespace:
        out = re.sub(r"\s+", " ", out).strip()
    return out
