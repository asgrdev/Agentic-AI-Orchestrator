from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class FrameQuality:
    mean_brightness: float
    blur_score: float
    motion_score: float
    occlusion_score: float

@dataclass
class Detection:
    frame_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    track_id: Optional[int] = None

@dataclass
class SegmentationMask:
    frame_id: int
    track_id: Optional[int]
    class_name: str
    confidence: float
    rle: Optional[str] = None
    area_ratio: Optional[float] = None

@dataclass
class PoseKeypoints:
    frame_id: int
    track_id: Optional[int]
    confidence: float
    keypoints: List[Tuple[float, float, float]]
    pose_quality: float

@dataclass
class DetectorStats:
    person_count_mean: float
    person_count_max: int
    object_histogram: Dict[str, int]

@dataclass
class DetectorResult:
    model_name: str
    enabled: bool
    conf_threshold: float
    iou_threshold: float
    detections: List[Detection]
    stats: DetectorStats
    confidence: float

@dataclass
class SegmentationResult:
    model_name: str
    enabled: bool
    masks: List[SegmentationMask]
    confidence: float

@dataclass
class PoseResult:
    model_name: str
    enabled: bool
    poses: List[PoseKeypoints]
    confidence: float

@dataclass
class SceneResult:
    enabled: bool
    label_level_1: str
    label_level_2: Optional[str]
    confidence: float
    features: Dict[str, float]

@dataclass
class EmbeddingResult:
    enabled: bool
    model_name: str
    dimension: int
    vector_path: Optional[str]
    l2_norm: Optional[float]
    confidence: float

@dataclass
class VisionSegPolicy:
    window_duration_sec: float
    fps_infer: float
    snapshot_every_sec: float
    detection_conf_threshold: float
    detection_iou_threshold: float
    tracker_enabled: bool
    track_ttl_sec: float
    enable_segmentation: bool
    enable_pose: bool

@dataclass
class VisionSegConfig:
    policy: VisionSegPolicy

@dataclass
class VisionSeg:
    seg_id: str
    source_id: str
    t0_ms: int
    t1_ms: int
    fps_infer: float
    frames_used: int

    quality: FrameQuality
    detector: DetectorResult
    segmentation: Optional[SegmentationResult] = None
    pose: Optional[PoseResult] = None
    scene: Optional[SceneResult] = None
    embedding: Optional[EmbeddingResult] = None

    snapshots: List[str] = field(default_factory=list)
    aligned_audio_seg_ids: List[str] = field(default_factory=list)
    graph_node_ids: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        def convert(obj):
            if obj is None:
                return None
            if hasattr(obj, "__dict__"):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        return convert(self)
