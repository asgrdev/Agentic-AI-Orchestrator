from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
from pathlib import Path
from .base import TextSource, TextPacket

@dataclass
class FileTextSource(TextSource):
    file_path: str
    source_id: str = "file_text"
    encoding: str = "utf-8"

    def __post_init__(self) -> None:
        self._lines = Path(self.file_path).read_text(encoding=self.encoding, errors="replace").splitlines()
        self._index = 0
        self._packet_id = 0

    def read(self) -> Optional[TextPacket]:
        if self._index >= len(self._lines):
            return None
        line = self._lines[self._index].rstrip("\n")
        self._index += 1
        self._packet_id += 1
        return TextPacket(
            packet_id=self._packet_id,
            created_at_ms=int(time.time() * 1000),
            text=line,
            source_type="file",
            source_id=self.source_id,
            confidence=None,
            meta={"line_index": self._index - 1, "file_path": self.file_path},
        )

    def close(self) -> None:
        return
