from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import platform
import time
from .base import VideoSource, VideoFrame
from ..cv2_compat import cv2

logger = logging.getLogger(__name__)


@dataclass
class CameraVideoSource(VideoSource):
    camera_index: int = 0
    source_id: str = "cam0"
    backend: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    probe_limit: int = 6

    def __post_init__(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is required. Install: pip install opencv-python")

        self._cap, used_backend = self._open_camera_with_fallback(self.camera_index)
        if self._cap is None:
            available = self.list_available_cameras(max_index=max(1, int(self.probe_limit)))
            raise RuntimeError(
                f"Could not open camera index {self.camera_index} on {platform.system()}. "
                f"Probed cameras: {available}"
            )
        if used_backend is not None:
            logger.info("Camera %s opened with backend=%s", self.camera_index, used_backend)

        self._apply_capture_settings()
        self._frame_id = 0

    @staticmethod
    def _backend_candidates(preferred: Optional[int]) -> List[Optional[int]]:
        if cv2 is None:
            return []
        candidates: List[Optional[int]] = []
        if preferred is not None:
            candidates.append(preferred)

        system = platform.system().lower()
        if "darwin" in system or "mac" in system:
            if hasattr(cv2, "CAP_AVFOUNDATION"):
                candidates.append(cv2.CAP_AVFOUNDATION)
        elif "linux" in system:
            if hasattr(cv2, "CAP_V4L2"):
                candidates.append(cv2.CAP_V4L2)
        elif "windows" in system:
            if hasattr(cv2, "CAP_DSHOW"):
                candidates.append(cv2.CAP_DSHOW)
            if hasattr(cv2, "CAP_MSMF"):
                candidates.append(cv2.CAP_MSMF)

        candidates.append(None)  # CAP_ANY fallback
        # preserve order, remove duplicates
        dedup: List[Optional[int]] = []
        for x in candidates:
            if x not in dedup:
                dedup.append(x)
        return dedup

    def _open_camera_with_fallback(self, camera_index: int) -> Tuple[Optional["cv2.VideoCapture"], Optional[int]]:
        for backend in self._backend_candidates(self.backend):
            try:
                cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    continue
                return cap, backend
            except Exception:
                continue
        return None, None

    def _apply_capture_settings(self) -> None:
        if self._cap is None:
            return
        if self.width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))

    @classmethod
    def list_available_cameras(cls, max_index: int = 6, preferred_backend: Optional[int] = None) -> List[int]:
        if cv2 is None:
            return []
        found: List[int] = []
        for i in range(max(1, int(max_index))):
            # Probe manually to avoid allocating a full source object per index.
            for backend in cls._backend_candidates(preferred_backend):
                try:
                    cap = cv2.VideoCapture(i) if backend is None else cv2.VideoCapture(i, backend)
                    if not cap.isOpened():
                        cap.release()
                        continue
                    ok, frame = cap.read()
                    cap.release()
                    if ok and frame is not None:
                        found.append(i)
                        break
                except Exception:
                    continue
        return found

    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        # timeout is accepted for API compatibility with poll-based callers.
        _ = timeout
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._frame_id += 1
        return VideoFrame(frame_id=self._frame_id, t_ms=int(time.time()*1000), bgr=frame, source_id=self.source_id)

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
