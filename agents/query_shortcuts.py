"""
میان‌برهای تشخیص query — بدون وابستگی سنگین (قابل تست مستقل).

مشکل: کوئری‌های ساده مثل «hello» کل پایپ‌لاین RAG را طی می‌کردند
(phi3 ≈ ۴۸ ثانیه + retrieve + refresh + granite) در حالی که مدل می‌تواند
مستقیم جواب بدهد. این ماژول دو میان‌بر ارزان (regex، بدون LLM) می‌دهد:

- is_conversational(query): سلام/احوال‌پرسی/تشکر/گفت‌وگوی کوتاه → پاسخ مستقیم
- extract_calc_expression(query): عبارت محاسباتی → اجرای مستقیم skill «calculate»
"""
from __future__ import annotations

import re
from typing import Optional

# عبارت‌های گفت‌وگویی که پاسخ‌شان نیاز به retrieval ندارد (en + fa)
_CONVERSATIONAL_EXACT = {
    # greetings
    "hi", "hello", "hey", "yo", "hi there", "hello there", "good morning",
    "good afternoon", "good evening", "good night",
    "سلام", "درود", "سلام علیکم", "صبح بخیر", "عصر بخیر", "شب بخیر", "ظهر بخیر",
    # farewell
    "bye", "goodbye", "see you", "خداحافظ", "بدرود", "فعلا", "خدانگهدار",
    # thanks / ack
    "thanks", "thank you", "thx", "ok", "okay", "cool", "great", "nice",
    "مرسی", "ممنون", "متشکرم", "تشکر", "باشه", "اوکی", "عالی", "دمت گرم",
    # small talk / identity
    "how are you", "how are you?", "what's up", "whats up", "who are you",
    "who are you?", "what can you do", "what can you do?",
    "خوبی", "خوبی؟", "چطوری", "چطوری؟", "حالت چطوره", "حالت چطوره؟",
    "چه خبر", "چه خبر؟", "تو کی هستی", "تو کی هستی؟", "چیکار میتونی بکنی",
    # test
    "test", "تست", "testing",
}

_GREETING_PREFIXES = (
    "hi ", "hello ", "hey ", "سلام ", "درود ",
)


def _normalize(query: str) -> str:
    q = query.strip().lower()
    q = re.sub(r"[!.،ـ]+$", "", q).strip()
    return q


def is_conversational(query: str) -> bool:
    """
    تشخیص محافظه‌کارانه‌ی گفت‌وگوی ساده — فقط وقتی True که مطمئنیم query
    نیاز اطلاعاتی ندارد؛ در حالت شک، False (مسیر کامل RAG).
    """
    q = _normalize(query)
    if not q:
        return False

    if q in _CONVERSATIONAL_EXACT:
        return True

    # «سلام، خوبی؟» و ترکیب‌های کوتاهِ سلام + احوال‌پرسی
    if len(q.split()) <= 4 and q.startswith(_GREETING_PREFIXES):
        rest = q.split(" ", 1)[1] if " " in q else ""
        rest = re.sub(r"[?؟,]", " ", rest).strip()
        if not rest or all(
            w in _CONVERSATIONAL_EXACT or w in {"there", "دوست", "رفیق"}
            for w in rest.split()
        ):
            return True

    return False


# کلمات دستوری که قبل از عبارت ریاضی می‌آیند و باید حذف شوند
_CALC_LEAD_WORDS = re.compile(
    r"^(?:please\s+)?(?:calculate|compute|solve|eval(?:uate)?|"
    r"محاسبه(?:\s*کن)?|حساب(?:\s*کن)?|چقدر\s*(?:می‌شود|میشود|میشه)?)\s*[:،]?\s*",
    re.IGNORECASE,
)

# «X% of Y» / «X درصد Y» / «X% از Y»
_PERCENT_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(?:%|٪|درصد)\s*(?:of|از)?\s*([\d,]+(?:\.\d+)?)\s*(?:چقدر\s*(?:است|میشه|می‌شود)?|\?|؟)?$",
    re.IGNORECASE,
)

# عبارت ریاضی خالص: فقط عدد/عملگر/پرانتز
_PURE_EXPR_RE = re.compile(r"^[\d\s\.\+\-\*/\(\)\^%,]+$")


def extract_calc_expression(query: str) -> Optional[str]:
    """
    اگر query یک محاسبه‌ی ساده باشد، عبارت قابل‌اجرا برای skill «calculate»
    برمی‌گرداند؛ وگرنه None (→ مسیر عادی planner).
    """
    q = _normalize(query)
    q = re.sub(r"(چند|چقدر)\s*(است|میشه|می‌شود)?\s*[؟?]?$", "", q).strip()
    q = re.sub(r"[?؟=]+$", "", q).strip()
    q = _CALC_LEAD_WORDS.sub("", q).strip()
    if not q:
        return None

    m = _PERCENT_RE.match(q)
    if m:
        pct, value = m.group(1), m.group(2).replace(",", "")
        return f"({pct} / 100) * {value}"

    if _PURE_EXPR_RE.match(q):
        expr = q.replace(",", "").replace("^", "**").strip()
        # باید حداقل یک عملگر بین دو عدد باشد؛ «۱۲۳» به‌تنهایی محاسبه نیست
        if re.search(r"\d\s*(?:\*\*|[\+\-\*/])\s*[\d\(]", expr) and not expr.endswith(tuple("+-*/")):
            return expr

    return None
