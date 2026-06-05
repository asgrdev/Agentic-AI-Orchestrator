from __future__ import annotations
def guess_language(text: str) -> str:
    if not text:
        return "unknown"
    persian = 0
    latin = 0
    for ch in text:
        o = ord(ch)
        if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
            persian += 1
        elif (0x0041 <= o <= 0x007A) or (0x00C0 <= o <= 0x024F):
            latin += 1
    if persian >= max(2, latin):
        return "fa"
    if latin >= max(2, persian):
        return "en"
    return "unknown"
