from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol

@dataclass
class TextPacket:
    packet_id: int
    created_at_ms: int
    text: str
    source_type: str
    source_id: str
    t0_ms: Optional[int] = None
    t1_ms: Optional[int] = None
    confidence: Optional[float] = None
    meta: dict | None = None

class TextSource(Protocol):
    def read(self) -> Optional[TextPacket]: ...
    def close(self) -> None: ...
