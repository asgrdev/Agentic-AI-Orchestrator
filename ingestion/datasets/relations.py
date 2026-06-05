from dataclasses import dataclass
from typing import Optional


@dataclass
class Relation:

    subject: str
    subject_id: Optional[str]

    predicate: str

    object: str
    object_id: Optional[str]

    confidence: float

    evidence: Optional[str] = None

    source_doc: Optional[str] = None
    source_chunk: Optional[str] = None

