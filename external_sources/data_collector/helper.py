import hashlib
import re
import asyncio
import time
from datetime import datetime
from typing import Any, Callable, TypeVar,List
from functools import wraps
from loguru import logger

T = TypeVar("T")
import tiktoken

class AdvancedChunker:

    def __init__(
        self,
        model_name: str = "gpt-4o",  # یا هر مدلی که استفاده می‌کنید
        max_tokens: int = 400,
        overlap_sentences: int = 2,
        min_chunk_tokens: int = 50,
        chunk_delimiter: str = "\n\n" # برای جدا کردن chunkها در خروجی نهایی
    ):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.min_chunk_tokens = min_chunk_tokens
        self.chunk_delimiter = chunk_delimiter

    def _get_token_count(self, text: str) -> int:
        """شمارش دقیق توکن با استفاده از tiktoken."""
        return len(self.encoding.encode(text))

    def _split_sentences(self, text: str) -> List[str]:
        """تقسیم متن به جمله‌ها با حفظ ساختار (مثلاً حفظ نقطه در پایان جمله)."""
        # این regex سعی می‌کند تا حد امکان نقاط و پایان جمله‌ها را حفظ کند
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sent_tokens = self._get_token_count(sentence)

            # بررسی اگر اضافه شدن جمله فعلی از حد تجاوز کند
            if current_tokens + sent_tokens > self.max_tokens:
                # ایجاد chunk فعلی اگر حداقل اندازه را داشته باشد
                if current_tokens >= self.min_chunk_tokens:
                    chunks.append(self.chunk_delimiter.join(current_chunk_sentences))

                # مدیریت overlap: جملات آخر را نگه دارید
                # نیاز داریم که جملات قبلی را دوباره token count کنیم
                # این بخش کمی محاسباتی است، ولی برای overlap دقیق لازم است
                overlap_chunk_sentences = current_chunk_sentences[-self.overlap_sentences:]
                overlap_text = self.chunk_delimiter.join(overlap_chunk_sentences)
                current_tokens = self._get_token_count(overlap_text)
                current_chunk_sentences = overlap_chunk_sentences

                # اگر حتی با overlap هم جمله فعلی بزرگتر از max_tokens باشد، آن را جداگانه chunk کنید
                # یا یک chunk از خود جمله بسازید (با ریسک بزرگ شدن chunk)
                if sent_tokens > self.max_tokens:
                    # در این حالت، معمولا جمله را split می‌کنند یا آن را به عنوان یک chunk بزرگ قبول می‌کنند
                    # اینجا برای سادگی، آن را به عنوان chunk قبول می‌کنیم و reset می‌کنیم
                    if current_tokens >= self.min_chunk_tokens:
                         chunks.append(self.chunk_delimiter.join(current_chunk_sentences))

                    # حالا جمله فعلی را به عنوان یک chunk جدید در نظر بگیریم
                    current_chunk_sentences = [sentence]
                    current_tokens = sent_tokens
                    # اینجا می‌توانیم اگر این chunk هم بزرگ بود، هشدار دهیم یا آن را chunk کنیم
                    if current_tokens >= self.max_tokens: # اگر خود جمله بزرگ است
                        chunks.append(self.chunk_delimiter.join(current_chunk_sentences))
                        current_chunk_sentences = [] # reset
                        current_tokens = 0
                    continue # برو سراغ جمله بعدی

            current_chunk_sentences.append(sentence)
            current_tokens += sent_tokens

        # اضافه کردن آخرین chunk اگر وجود دارد
        if current_chunk_sentences and current_tokens >= self.min_chunk_tokens:
            chunks.append(self.chunk_delimiter.join(current_chunk_sentences))

        return chunks


class DataHelper:
    """ابزارهای کمکی یکپارچه"""

    # ─── Text Processing ────────────────────────────────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        """پاک‌سازی متن از کاراکترهای اضافه"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)           # فضاهای اضافه
        text = re.sub(r'\n{3,}', '\n\n', text)     # خطوط خالی اضافه
        text = re.sub(r'[^\S\r\n]+', ' ', text)    # tab و space ترکیبی
        return text.strip()

    @staticmethod
    def truncate(text: str, max_chars: int = 2000) -> str:
        """کوتاه کردن متن"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> list[str]:
        """استخراج ساده کلیدواژه بدون نیاز به NLP"""
        stop_words = {
            "the", "a", "an", "is", "in", "of", "and", "or",
            "to", "for", "with", "that", "this", "it", "as",
            "at", "be", "by", "from", "on", "was", "are"
        }
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        freq: dict[str, int] = {}
        for word in words:
            if word not in stop_words:
                freq[word] = freq.get(word, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:top_n]

    @staticmethod
    def detect_language(text: str) -> str:
        """تشخیص ساده زبان"""
        persian_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        arabic_chars  = len(re.findall(r'[\u0600-\u06FF]', text))
        if persian_chars > len(text) * 0.3:
            return "fa"
        if arabic_chars > len(text) * 0.3:
            return "ar"
        return "en"

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> list[str]:
        """تقسیم متن به chunk با overlap"""
        # words = text.split()
        # chunks = []
        # step = max(chunk_size - overlap, 1)
        # for i in range(0, len(words), step):
        #     chunk = " ".join(words[i:i + chunk_size])
        #     if len(chunk.strip()) > 20:
        #         chunks.append(chunk)
        chunks= AdvancedChunker(overlap_sentences=1).chunk_text(text=text)
        return chunks

    # ─── ID Generation ──────────────────────────────────────────────────

    @staticmethod
    def make_id(source: str, identifier: str) -> str:
        """ساخت ID یکتا"""
        raw = f"{source}:{identifier}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ─── Rate Limiting ───────────────────────────────────────────────────

    @staticmethod
    def retry(
        max_attempts: int = 3,
        delay: float = 2.0,
        backoff: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """دکوراتور retry با exponential backoff"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                current_delay = delay
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_error = e
                        if attempt < max_attempts:
                            logger.warning(
                                f"⚠️  {func.__name__} attempt {attempt} failed: {e}"
                                f" — retry in {current_delay:.1f}s"
                            )
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            logger.error(f"❌ {func.__name__} failed after {max_attempts} attempts")
                raise last_error
            return wrapper
        return decorator

    @staticmethod
    def async_retry(
        max_attempts: int = 3,
        delay: float = 2.0,
        backoff: float = 2.0,
    ):
        """دکوراتور async retry"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                current_delay = delay
                last_error = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        if attempt < max_attempts:
                            logger.warning(f"⚠️  attempt {attempt} failed: {e}")
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff
                raise last_error
            return wrapper
        return decorator

    # ─── Data Normalization ─────────────────────────────────────────────

    @staticmethod
    def normalize_date(value: Any) -> datetime | None:
        """تبدیل انواع تاریخ به datetime"""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            formats = [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d/%m/%Y",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return None

    @staticmethod
    def safe_get(data: dict, *keys, default=None) -> Any:
        """دسترسی امن به کلیدهای تودرتو"""
        for key in keys:
            if not isinstance(data, dict):
                return default
            data = data.get(key, default)
        return data

    # ─── Display ────────────────────────────────────────────────────────

    @staticmethod
    def print_result_summary(result) -> None:
        """نمایش خلاصه نتیجه"""
        status = "✅" if result.success else "❌"
        logger.info(
            f"{status} [{result.source}] "
            f"total={result.total} | "
            f"errors={len(result.errors)} | "
            f"time={result.elapsed_sec:.2f}s"
        )
        for err in result.errors:
            logger.warning(f"   ⚠️  {err}")
