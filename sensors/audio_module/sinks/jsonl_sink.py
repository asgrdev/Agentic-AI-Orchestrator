from __future__ import annotations
from dataclasses import dataclass
import json
import os
from ..schema.audioseg import AudioSeg

@dataclass
class JsonlSink:
    output_jsonl_path: str

    def __post_init__(self) -> None:
        os.makedirs(os.path.dirname(self.output_jsonl_path) or ".", exist_ok=True)

    def write(self, seg: AudioSeg) -> None:
        with open(self.output_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(seg.to_json_dict(), ensure_ascii=False) + "\n")
