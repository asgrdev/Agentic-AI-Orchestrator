from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import queue, time
from .base import TextSource, TextPacket

@dataclass
class PushTextSource(TextSource):
    source_id: str = "push_text"
    source_type: str = "user_prompt"
    max_queue: int = 500

    def __post_init__(self) -> None:
        self._queue: "queue.Queue[TextPacket]" = queue.Queue(maxsize=self.max_queue)
        self._packet_id = 0

    def push(self, text: str, confidence: Optional[float] = None, t0_ms: Optional[int] = None, t1_ms: Optional[int] = None, meta: Optional[dict] = None) -> None:
        self._packet_id += 1
        pkt = TextPacket(
            packet_id=self._packet_id,
            created_at_ms=int(time.time() * 1000),
            text=text,
            source_type=self.source_type,
            source_id=self.source_id,
            t0_ms=t0_ms,
            t1_ms=t1_ms,
            confidence=confidence,
            meta=meta or {},
        )
        try:
            self._queue.put_nowait(pkt)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(pkt)
            except queue.Full:
                pass

    def read(self) -> Optional[TextPacket]:
        try:
            return self._queue.get(timeout=0.2)
        except queue.Empty:
            return None

    def close(self) -> None:
        return
