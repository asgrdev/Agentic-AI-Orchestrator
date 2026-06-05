from difflib import SequenceMatcher
import re


class EntityNormalizer:

    def normalize(self, text: str) -> str:

        text = text.lower()

        text = re.sub(r"[^\w\s]", "", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()


def similarity(a, b):

    return SequenceMatcher(None, a, b).ratio()


class EntityCanonicalizer:

    def __init__(self, threshold=0.85):

        self.normalizer = EntityNormalizer()

        self.threshold = threshold

        self.entities = {}
        self.canonical = {}

    def canonicalize(self, entity):

        norm = self.normalizer.normalize(entity.text)

        for existing_norm, canonical in self.entities.items():

            sim = similarity(norm, existing_norm)

            if sim > self.threshold:

                self.canonical[entity.text] = canonical

                return canonical

        self.entities[norm] = entity.text

        self.canonical[entity.text] = entity.text

        return entity.text
