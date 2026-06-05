from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class Track:
    track_id: int
    class_name: str
    bbox_xyxy: Tuple[float, float, float, float]
    last_seen_ms: int
    frames_seen: int = 1

def iou_xyxy(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    area_a = max(0.0, ax2-ax1)*max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1)*max(0.0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-9)

@dataclass
class SimpleIouTracker:
    iou_match_threshold: float = 0.30
    track_ttl_sec: float = 3.0

    def __post_init__(self) -> None:
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[tuple[str, float, Tuple[float,float,float,float]]], now_ms: int) -> List[tuple[int|None, str, float, Tuple[float,float,float,float]]]:
        ttl_ms = int(self.track_ttl_sec * 1000)
        for tid in list(self._tracks.keys()):
            if (now_ms - self._tracks[tid].last_seen_ms) > ttl_ms:
                del self._tracks[tid]

        out = []
        for cls, conf, bbox in detections:
            best_tid, best_iou = None, 0.0
            for tid, tr in self._tracks.items():
                if tr.class_name != cls:
                    continue
                s = iou_xyxy(tr.bbox_xyxy, bbox)
                if s > best_iou:
                    best_iou, best_tid = s, tid
            if best_tid is not None and best_iou >= self.iou_match_threshold:
                tr = self._tracks[best_tid]
                tr.bbox_xyxy = bbox
                tr.last_seen_ms = now_ms
                tr.frames_seen += 1
                out.append((best_tid, cls, conf, bbox))
            else:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = Track(track_id=tid, class_name=cls, bbox_xyxy=bbox, last_seen_ms=now_ms)
                out.append((tid, cls, conf, bbox))
        return out
