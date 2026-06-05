from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GatingDecision:
    run_segmentation: bool
    run_pose: bool

class VisionPolicy:
    def __init__(self, enable_segmentation: bool, enable_pose: bool, seg_requires_person: bool, pose_requires_person: bool) -> None:
        self.enable_segmentation = enable_segmentation
        self.enable_pose = enable_pose
        self.seg_requires_person = seg_requires_person
        self.pose_requires_person = pose_requires_person

    def decide(self, person_present: bool) -> GatingDecision:
        run_seg = self.enable_segmentation and (person_present if self.seg_requires_person else True)
        run_pose = self.enable_pose and (person_present if self.pose_requires_person else True)
        return GatingDecision(run_segmentation=run_seg, run_pose=run_pose)
