"""
استخراج «توکن‌های فکر» از خروجی خام LLM ها.

مدل‌های reasoning (DeepSeek-R1، Qwen-thinking، granite در حالت CoT و ...)
فکرشان را داخل تگ‌هایی مثل <think>...</think> یا بخش REASONING: می‌نویسند.
این ماژول آن بخش را از پاسخ نهایی جدا می‌کند تا:
  1. پاسخ تمیز به کاربر برسد (بدون تگ)
  2. فرایند فکر جداگانه در UI (حباب «🧠 فرایند فکر») نمایش داده شود

استفاده:
    thinking, answer = split_thinking(raw_output)
"""
from __future__ import annotations

import re

# تگ‌های رایج فکر در مدل‌های مختلف — به‌ترتیب امتحان می‌شوند
_TAG_PATTERNS = [
    re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|thinking\|>(.*?)<\|/thinking\|>", re.DOTALL),
    re.compile(r"◁think▷(.*?)◁/think▷", re.DOTALL),
]

# تگ فکرِ بازمانده (مدل قبل از بستن تگ قطع شده) — همه‌ی متن بعدش فکر است
_OPEN_TAG = re.compile(r"<think(?:ing)?>", re.IGNORECASE)

# الگوی REASONING:/ANSWER: که prompts.py خودمان از مدل می‌خواهد
_REASONING_ANSWER = re.compile(
    r"REASONING\s*:\s*(.*?)\s*ANSWER\s*:\s*(.*)", re.DOTALL | re.IGNORECASE
)


def split_thinking(text: str) -> tuple[str, str]:
    """
    جداسازی (thinking, answer) از خروجی خام مدل.

    اگر هیچ بلاک فکری پیدا نشود: ("", متن کامل).
    چند بلاک فکر → با خط جدید به هم می‌چسبند.
    """
    if not text:
        return "", text

    thinking_parts: list[str] = []
    answer = text

    # ۱. تگ‌های کامل <think>...</think>
    for pat in _TAG_PATTERNS:
        matches = pat.findall(answer)
        if matches:
            thinking_parts.extend(m.strip() for m in matches if m.strip())
            answer = pat.sub("", answer)

    # ۲. تگ باز بدون بسته‌شدن — بقیه‌ی متن فکر است، پاسخی نمانده
    m = _OPEN_TAG.search(answer)
    if m:
        rest = answer[m.end():].strip()
        if rest:
            thinking_parts.append(rest)
        answer = answer[: m.start()]

    # ۳. الگوی REASONING:/ANSWER: (اگر هنوز در پاسخ مانده باشد)
    m = _REASONING_ANSWER.search(answer)
    if m:
        if m.group(1).strip():
            thinking_parts.append(m.group(1).strip())
        answer = answer[: m.start()] + m.group(2)

    return "\n\n".join(thinking_parts).strip(), answer.strip()


def strip_thinking(text: str) -> str:
    """فقط پاسخ تمیز — فکر دور ریخته می‌شود"""
    return split_thinking(text)[1]
