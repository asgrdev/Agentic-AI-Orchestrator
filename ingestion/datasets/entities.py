from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Entity:

    text: str
    type: str
    start: int
    end: int
    confidence: float

    normalized: Optional[str] = None

    linked_id: Optional[str] = None
    kb_url: Optional[str] = None
    wikipedia_url: Optional[str] = None
    description: Optional[str] = None

    aliases: list[str] = field(default_factory=list)
    label: str = ""               # alias برای type

    source_doc: Optional[str] = None
    source_chunk: Optional[str] = None

    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.text.lower().strip()
        if not self.id:
            self.id = self.normalized.replace(" ", "_")
        if not self.canonical:
            self.canonical = self.text
        if not self.label:
            self.label = self.type
