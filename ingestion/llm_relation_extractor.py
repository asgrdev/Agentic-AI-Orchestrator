import json

from ingestion.datasets.relations import Relation
#============================================================
from mlx_lm import load, generate


class MLXLLM:

    def __init__(self, model_path):

        self.model, self.tokenizer = load(model_path)

    def complete(self, prompt, max_tokens=512):

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=0.1
        )

        return response

#============================================================ 
RELATION_PROMPT = """
You are an information extraction system.

Extract factual relations between entities from the text.

Rules:
- Only use the provided entities
- Do not invent entities
- Output ONLY JSON
- Each relation must have: subject, predicate, object, confidence

Entities:
{entities}

Text:
{text}

Return format:

[
  {{
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "confidence": 0.0-1.0
  }}
]
"""

class LLMRelationExtractor:

    def __init__(self, config):

        model = config.get("model_name")

        self.llm = MLXLLM(model)

    def extract(self, text, entities):

        entity_names = [e.text for e in entities]

        prompt = RELATION_PROMPT.format(
            text=text,
            entities=entity_names
        )

        response = self.llm.complete(prompt)

        try:

            data = json.loads(self._extract_json(response))

        except Exception:

            return []

        relations = []

        for r in data:

            subj = self._find_entity(r["subject"], entities)
            obj = self._find_entity(r["object"], entities)

            if not subj or not obj:
                continue

            relations.append(
                Relation(
                    subject=subj.text,
                    subject_id=subj.linked_id,
                    predicate=r["predicate"],
                    object=obj.text,
                    object_id=obj.linked_id,
                    confidence=r.get("confidence", 0.7)
                )
            )

        return relations

    def _extract_json(self, text):

        start = text.find("[")
        end = text.rfind("]")

        return text[start:end+1]

    def _find_entity(self, name, entities):

        name = name.lower()

        for e in entities:

            if e.text.lower() == name:
                return e

        return None
