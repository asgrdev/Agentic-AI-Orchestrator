from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import time
from .base import VideoSource, VideoFrame
from ..cv2_compat import cv2

@dataclass
class RtspVideoSource(VideoSource):
    rtsp_url: str
    source_id: str = "rtsp0"

    def __post_init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required. Install: pip install opencv-python")
        self._cap = cv2.VideoCapture(self.rtsp_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {self.rtsp_url}")
        self._frame_id = 0

    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        _ = timeout
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._frame_id += 1
        return VideoFrame(frame_id=self._frame_id, t_ms=int(time.time()*1000), bgr=frame, source_id=self.source_id)

    def close(self) -> None:
        try: self._cap.release()
        except Exception: pass
