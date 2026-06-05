from ingestion.relation_ontology import RELATION_ONTOLOGY


class RelationNormalizer:

    def __init__(self):

        self.map = {}

        for canonical, variants in RELATION_ONTOLOGY.items():

            for v in variants:
                self.map[v.lower()] = canonical

    def normalize(self, predicate):

        p = predicate.lower().strip()

        if p in self.map:
            return self.map[p]

        return predicate.upper()
