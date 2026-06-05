from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

@dataclass(frozen=True)
class RuntimeConfig:
    window_duration_sec: float = 15.0
    fps_infer: float = 3.0

    detection_conf_threshold: float = 0.25
    detection_iou_threshold: float = 0.45
    max_detections_per_frame: int = 200

    enable_segmentation: bool = True
    enable_pose: bool = True
    segmentation_requires_person: bool = True
    pose_requires_person: bool = True

    tracker_enabled: bool = True
    tracker_iou_match_threshold: float = 0.30
    track_ttl_sec: float = 3.0

    scene_classifier_enabled: bool = True

    embedding_enabled: bool = True
    embedding_dim: int = 512

    snapshot_every_sec: float = 5.0
    snapshot_jpeg_quality: int = 90

    # Optional local datasets used by training/analysis flows.
    dataset_resources: Dict[str, str] = field(default_factory=dict)
