import spacy
from typing import List

from ingestion.datasets.relations import Relation
from ingestion.datasets.entities import Entity


class RelationExtractor:

    def __init__(self):

        self.nlp = spacy.load("en_core_web_trf")

    def extract(self, text: str, entities: List[Entity]) -> List[Relation]:

        doc = self.nlp(text)

        relations = []

        for token in doc:

            if token.pos_ == "VERB":

                subject = None
                obj = None

                for child in token.children:

                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child

                    if child.dep_ in ("dobj", "pobj", "attr"):
                        obj = child

                if subject and obj:

                    subj_ent = self._match_entity(subject, entities)
                    obj_ent = self._match_entity(obj, entities)

                    if subj_ent and obj_ent:

                        relations.append(
                            Relation(
                                subject=subj_ent.text,
                                subject_id=subj_ent.linked_id,
                                predicate=token.lemma_,
                                object=obj_ent.text,
                                object_id=obj_ent.linked_id,
                                confidence=0.7
                            )
                        )

        return relations

    def _match_entity(self, token, entities):

        for e in entities:

            if token.idx >= e.start and token.idx <= e.end:
                return e

        return None
