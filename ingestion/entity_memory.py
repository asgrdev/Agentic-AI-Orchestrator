"""
EntityMemory - نگهداری حافظه sliding window از entities در طول پردازش متن
برای استفاده در ContextRelationExtractor
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EntityRecord:
    """یک entity با متادیتا"""
    entity_id: str
    text: str
    entity_type: str
    confidence: float = 1.0
    sentence_index: int = 0
    metadata: dict = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.entity_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EntityRecord):
            return False
        return self.entity_id == other.entity_id


class EntityMemory:
    """
    حافظه sliding window برای entities در متن.

    کاربرد:
        در پردازش جمله‌به‌جمله، entities جملات قبلی را
        به عنوان context برای استخراج روابط فراهم می‌کند.

    مثال:
        "Alice works at OpenAI. She founded it in 2015."
        → جمله دوم نیاز دارد بداند "She" = Alice (از حافظه)
    """

    def __init__(
        self,
        window_size: int = 5,
        max_entity_age: int = 3,
        decay_factor: float = 0.8,
    ) -> None:
        """
        Args:
            window_size:     حداکثر تعداد جملات در حافظه
            max_entity_age:  حداکثر سن entity (تعداد جملات) قبل از حذف
            decay_factor:    ضریب کاهش confidence با هر جمله
        """
        self.window_size = window_size
        self.max_entity_age = max_entity_age
        self.decay_factor = decay_factor

        # deque از list[EntityRecord] - هر عنصر = entities یک جمله
        self._window: deque[list[EntityRecord]] = deque(maxlen=window_size)
        self._current_sentence_index: int = 0

        # index سریع برای جستجو بر اساس text
        self._text_index: dict[str, EntityRecord] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add(self, entities: list[Any]) -> None:
        """
        اضافه کردن entities یک جمله به حافظه.

        Args:
            entities: لیست Entity objects (باید دارای .text و .entity_type باشند)
        """
        if not entities:
            self._window.append([])
            self._current_sentence_index += 1
            return

        records = []
        for e in entities:
            record = self._to_record(e, self._current_sentence_index)
            records.append(record)
            # آپدیت text index با آخرین نسخه
            self._text_index[record.text.lower()] = record

        self._window.append(records)
        self._current_sentence_index += 1

        # پاکسازی text_index از entities قدیمی
        self._cleanup_text_index()

    def get_context_entities(
        self,
        max_age: Optional[int] = None,
        min_confidence: float = 0.0,
        entity_types: Optional[list[str]] = None,
    ) -> list[Any]:
        """
        دریافت entities موجود در حافظه به عنوان context.

        Args:
            max_age:          حداکثر سن (None = استفاده از max_entity_age)
            min_confidence:   حداقل confidence
            entity_types:     فیلتر بر اساس نوع entity

        Returns:
            لیست از Entity-like objects با confidence decay اعمال شده
        """
        effective_max_age = max_age if max_age is not None else self.max_entity_age
        result: list[EntityRecord] = []
        seen_ids: set[str] = set()

        for age, sentence_entities in enumerate(reversed(self._window)):
            if age >= effective_max_age:
                break

            for record in sentence_entities:
                if record.entity_id in seen_ids:
                    continue

                # اعمال decay بر اساس سن
                decayed_conf = record.confidence * (self.decay_factor ** age)

                if decayed_conf < min_confidence:
                    continue

                if entity_types and record.entity_type not in entity_types:
                    continue

                # ساخت کپی با confidence به‌روز شده
                aged_record = EntityRecord(
                    entity_id=record.entity_id,
                    text=record.text,
                    entity_type=record.entity_type,
                    confidence=decayed_conf,
                    sentence_index=record.sentence_index,
                    metadata={**record.metadata, "age": age},
                )
                result.append(aged_record)
                seen_ids.add(record.entity_id)

        return result

    def find_by_text(self, text: str) -> Optional[EntityRecord]:
        """جستجوی entity بر اساس text (case-insensitive)"""
        return self._text_index.get(text.lower())

    def clear(self) -> None:
        """پاکسازی کامل حافظه"""
        self._window.clear()
        self._text_index.clear()
        self._current_sentence_index = 0

    def __len__(self) -> int:
        return sum(len(s) for s in self._window)

    def __repr__(self) -> str:
        return (
            f"EntityMemory("
            f"window={len(self._window)}/{self.window_size}, "
            f"entities={len(self)}, "
            f"sentence={self._current_sentence_index})"
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _to_record(self, entity: Any, sentence_index: int) -> EntityRecord:
        """تبدیل entity object به EntityRecord"""
        # پشتیبانی از چند نوع entity object
        if isinstance(entity, EntityRecord):
            return EntityRecord(
                entity_id=entity.entity_id,
                text=entity.text,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
                sentence_index=sentence_index,
                metadata=entity.metadata,
            )

        # اگر dataclass یا object با attributes باشد
        entity_id = (
            getattr(entity, "entity_id", None)
            or getattr(entity, "id", None)
            or f"{getattr(entity, 'entity_type', 'UNK')}_{getattr(entity, 'text', '')}"
        )
        return EntityRecord(
            entity_id=str(entity_id),
            text=getattr(entity, "text", str(entity)),
            entity_type=getattr(entity, "entity_type", "UNKNOWN"),
            confidence=getattr(entity, "confidence", 1.0),
            sentence_index=sentence_index,
            metadata=getattr(entity, "metadata", {}),
        )

    def _cleanup_text_index(self) -> None:
        """حذف entries قدیمی از text_index"""
        if len(self._window) < self.window_size:
            return

        # entities موجود در window فعلی
        active_ids: set[str] = set()
        for sentence_entities in self._window:
            for record in sentence_entities:
                active_ids.add(record.entity_id)

        # حذف از text_index هر چه دیگر active نیست
        stale_keys = [
            k for k, v in self._text_index.items()
            if v.entity_id not in active_ids
        ]
        for k in stale_keys:
            del self._text_index[k]
