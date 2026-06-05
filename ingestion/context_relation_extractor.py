from ingestion.entity_memory import EntityMemory
from ingestion.context_builder import ContextBuilder


class ContextRelationExtractor:

    def __init__(self, llm_extractor):

        self.llm_extractor = llm_extractor
        self.memory = EntityMemory()

        self.context_builder = ContextBuilder()

    def extract(self, text, entities):

        sentences = self.context_builder.split_sentences(text)

        relations = []

        for sent in sentences:

            context_entities = self.memory.get_context_entities()

            all_entities = entities + context_entities

            rels = self.llm_extractor.extract(
                sent,
                all_entities
            )

            relations.extend(rels)

            sentence_entities = [
                e for e in entities if e.text in sent
            ]

            self.memory.add(sentence_entities)

        return relations
