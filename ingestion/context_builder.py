"""
ContextBuilder - ساخت context غنی برای LLM relation extraction
شامل تقسیم جمله‌ها، فرمت‌بندی entities و ساخت prompt context
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SentenceSpan:
    """یک جمله با موقعیت در متن اصلی"""
    text: str
    start: int
    end: int
    index: int


class ContextBuilder:
    """
    ابزار ساخت context برای extraction مبتنی بر LLM.

    وظایف:
        1. تقسیم متن به جمله‌ها (sentence splitting)
        2. فرمت‌بندی entities برای prompt
        3. ساخت context window از چند جمله
        4. فرمت‌بندی graph context از KuzuDB
    """

    # ------------------------------------------------------------------ #
    # Sentence Splitting                                                   #
    # ------------------------------------------------------------------ #

    _SENT_BOUNDARY = re.compile(
        r"(?<!\w\.\w.)"          # جلوگیری از تقسیم روی "U.S.A"
        r"(?<![A-Z][a-z]\.)"    # جلوگیری از "Mr."
        r"(?<=\.|\?|!)"          # بعد از نقطه، ؟، !
        r"\s+(?=[A-Z\u0600-\u06FF])",  # فاصله + شروع با حرف بزرگ یا فارسی
        re.UNICODE,
    )

    def split_sentences(self, text: str) -> list[str]:
        """
        تقسیم متن به جمله‌ها.

        Args:
            text: متن ورودی

        Returns:
            لیست از جمله‌ها (بدون جمله خالی)
        """
        if not text or not text.strip():
            return []

        # نرمال‌سازی فاصله‌ها
        text = text.strip()

        # تلاش با spaCy اگر موجود باشد
        try:
            sentences = self._split_with_spacy(text)
            if sentences:
                return sentences
        except Exception:
            pass

        # fallback: regex-based splitting
        parts = self._SENT_BOUNDARY.split(text)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences if sentences else [text]

    def split_sentences_with_spans(self, text: str) -> list[SentenceSpan]:
        """
        تقسیم متن به جمله‌ها با موقعیت (برای citation).

        Returns:
            لیست SentenceSpan با start/end offset در متن اصلی
        """
        sentences = self.split_sentences(text)
        spans = []
        cursor = 0

        for idx, sent in enumerate(sentences):
            start = text.find(sent, cursor)
            if start == -1:
                start = cursor
            end = start + len(sent)
            spans.append(SentenceSpan(
                text=sent,
                start=start,
                end=end,
                index=idx,
            ))
            cursor = end

        return spans

    # ------------------------------------------------------------------ #
    # Entity Formatting                                                    #
    # ------------------------------------------------------------------ #

    def format_entities_for_prompt(
        self,
        entities: list[Any],
        include_confidence: bool = False,
    ) -> str:
        """
        فرمت‌بندی entities برای استفاده در LLM prompt.

        مثال خروجی:
            - Alice [PERSON] (conf: 0.95)
            - OpenAI [ORGANIZATION]
            - 2015 [DATE]
        """
        if not entities:
            return "No entities found."

        lines = []
        for e in entities:
            text = getattr(e, "text", str(e))
            etype = getattr(e, "entity_type", "UNKNOWN")
            conf = getattr(e, "confidence", None)

            line = f"- {text} [{etype}]"
            if include_confidence and conf is not None:
                line += f" (conf: {conf:.2f})"

            lines.append(line)

        return "\n".join(lines)

    def format_entities_inline(self, entities: list[Any]) -> str:
        """
        فرمت‌بندی فشرده entities (برای context کوتاه).

        مثال: "Alice(PERSON), OpenAI(ORG), 2015(DATE)"
        """
        if not entities:
            return ""

        parts = []
        for e in entities:
            text = getattr(e, "text", str(e))
            etype = getattr(e, "entity_type", "UNK")
            parts.append(f"{text}({etype})")

        return ", ".join(parts)

    # ------------------------------------------------------------------ #
    # Context Window Building                                              #
    # ------------------------------------------------------------------ #

    def build_context_window(
        self,
        sentences: list[str],
        current_index: int,
        window_before: int = 2,
        window_after: int = 1,
    ) -> str:
        """
        ساخت پنجره context از جملات اطراف جمله فعلی.

        Args:
            sentences:      تمام جملات متن
            current_index:  ایندکس جمله فعلی
            window_before:  تعداد جملات قبل
            window_after:   تعداد جملات بعد

        Returns:
            متن context با [CURRENT] مشخص شده
        """
        start = max(0, current_index - window_before)
        end = min(len(sentences), current_index + window_after + 1)

        parts = []
        for i in range(start, end):
            if i == current_index:
                parts.append(f"[CURRENT] {sentences[i]}")
            else:
                parts.append(f"[CONTEXT] {sentences[i]}")

        return " ".join(parts)

    # ------------------------------------------------------------------ #
    # Graph Context Formatting                                             #
    # ------------------------------------------------------------------ #

    def format_graph_context(
        self,
        nodes: list[dict],
        relations: list[dict],
        max_nodes: int = 20,
        max_relations: int = 30,
    ) -> str:
        """
        فرمت‌بندی context از KuzuDB برای LLM.

        Args:
            nodes:      لیست node dictionaries از KuzuDB
            relations:  لیست relation dictionaries از KuzuDB

        Returns:
            متن فرمت‌بندی شده برای prompt
        """
        lines = ["### Knowledge Graph Context\n"]

        # Entities
        if nodes:
            lines.append("**Entities:**")
            for node in nodes[:max_nodes]:
                name = node.get("name") or node.get("text", "Unknown")
                ntype = node.get("type", "Entity")
                props = {
                    k: v for k, v in node.items()
                    if k not in ("name", "text", "type", "embedding")
                    and v is not None
                }
                line = f"  - {name} [{ntype}]"
                if props:
                    prop_str = ", ".join(f"{k}={v}" for k, v in list(props.items())[:3])
                    line += f" ({prop_str})"
                lines.append(line)

            if len(nodes) > max_nodes:
                lines.append(f"  ... and {len(nodes) - max_nodes} more entities")

        # Relations
        if relations:
            lines.append("\n**Relations:**")
            for rel in relations[:max_relations]:
                src = rel.get("source") or rel.get("src", "?")
                tgt = rel.get("target") or rel.get("tgt", "?")
                rtype = rel.get("type") or rel.get("relation", "RELATED")
                conf = rel.get("confidence")

                line = f"  - {src} --[{rtype}]--> {tgt}"
                if conf is not None:
                    line += f" (conf: {conf:.2f})"
                lines.append(line)

            if len(relations) > max_relations:
                lines.append(f"  ... and {len(relations) - max_relations} more relations")

        return "\n".join(lines)

    def format_doc_context(
        self,
        search_results: list[Any],
        max_results: int = 5,
    ) -> str:
        """
        فرمت‌بندی نتایج vector search برای LLM.

        Args:
            search_results: لیست SearchResult از WeaviateStore

        Returns:
            متن فرمت‌بندی شده
        """
        if not search_results:
            return "No relevant documents found."

        lines = ["### Retrieved Documents\n"]

        for i, result in enumerate(search_results[:max_results], 1):
            # پشتیبانی از SearchResult dataclass و dict
            if hasattr(result, "content"):
                content = result.content
                score = getattr(result, "score", None)
                chunk_id = getattr(result, "chunk_id", f"doc_{i}")
                metadata = getattr(result, "metadata", {})
            else:
                content = result.get("content", result.get("text", ""))
                score = result.get("score")
                chunk_id = result.get("chunk_id", f"doc_{i}")
                metadata = result.get("metadata", {})

            lines.append(f"**[{i}] Source: {chunk_id}**")
            if score is not None:
                lines.append(f"   Score: {score:.4f}")

            # متادیتای مفید
            if metadata:
                useful_meta = {
                    k: v for k, v in metadata.items()
                    if k in ("source", "title", "date", "author", "page")
                    and v is not None
                }
                if useful_meta:
                    meta_str = ", ".join(f"{k}: {v}" for k, v in useful_meta.items())
                    lines.append(f"   Meta: {meta_str}")

            # محتوا (truncate اگر خیلی طولانی)
            content_preview = content[:500] + "..." if len(content) > 500 else content
            lines.append(f"   Content: {content_preview}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _split_with_spacy(self, text: str) -> list[str]:
        """تقسیم با spaCy اگر موجود باشد"""
        import spacy  # noqa: PLC0415
        # استفاده از مدل کوچک فقط برای sentencizer
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            nlp.add_pipe("sentencizer")
        except Exception:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")

        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
