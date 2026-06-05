from ingestion.relation_extractor import RelationExtractor
from ingestion.llm_relation_extractor import LLMRelationExtractor
from ingestion.relation_normalizer import RelationNormalizer
from ingestion.context_relation_extractor import ContextRelationExtractor


class RelationPipeline:

    def __init__(self, config):

        self.rule_extractor = RelationExtractor()
        self.llm_extractor = LLMRelationExtractor(config)
        self.context_extractor = ContextRelationExtractor(
            self.llm_extractor
        )

        self.normalizer = RelationNormalizer()
 
    def extract(self, chunk, entities):

        relations = self.context_extractor.extract(
            chunk["text"],
            entities
        )

        for r in relations:

            r.predicate = self.normalizer.normalize(
                r.predicate
            )

            r.source_chunk = chunk["id"]
            r.source_doc = chunk["title"]

        return relations
