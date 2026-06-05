from dataclasses import dataclass
import spacy
from gliner import GLiNER      # مدل NER بهتر برای domain-specific
from dataclasses import dataclass, field
from typing import Optional
from ingestion.datasets.entities import Entity


class NERExtractor:
    """
    ترکیب spaCy + GLiNER:
    - spaCy: موجودیت‌های عمومی سریع
    - GLiNER: موجودیت‌های domain-specific با label دلخواه
    """

    GLINER_LABELS = [
        "Person", "Organization", "Location",
        "Technology", "Concept", "Event",
        "Product", "Date", "Scientific Term",
        "Dataset",
        "Algorithm"
    ]

    def __init__(self, config: dict):
        self.spacy_nlp = spacy.load(
            config.get("spacy_model", "en_core_web_trf")
        )
        self.gliner = GLiNER.from_pretrained(
            config.get("gliner_model", "urchade/gliner_medium-v2.1")
        )
 
    def extract_batch(self, texts: list[str]) -> list[list[Entity]]:

        results = []

        for text in texts:
            results.append(self.extract(text))

        return results
    def extract(self, text: str) -> list[Entity]:
        """استخراج موجودیت با هر دو مدل + dedup"""
        spacy_ents  = self._extract_spacy(text)
        gliner_ents = self._extract_gliner(text)
        return self._deduplicate(spacy_ents + gliner_ents)

    def _extract_spacy(self, text: str) -> list[Entity]:
        doc = self.spacy_nlp(text)
        return [
            Entity(
                text=ent.text,
                type=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,
            )
            for ent in doc.ents
        ]

    def _extract_gliner(self, text: str) -> list[Entity]:
        predictions = self.gliner.predict_entities(
            text, self.GLINER_LABELS, threshold=0.5
        )
        return [
            Entity(
                text=p["text"],
                type=p["label"],
                start=p["start"],
                end=p["end"],
                confidence=p["score"],
            )
            for p in predictions
        ]

    def _deduplicate(self, entities: list[Entity]) -> list[Entity]:
        """حذف موجودیت‌های تکراری با span overlap"""
        entities.sort(key=lambda e: (e.start, -e.confidence))
        result, last_end = [], -1
        for ent in entities:
            if ent.start >= last_end:
                result.append(ent)
                last_end = ent.end
        return result

