"""VisionSeg module: YOLO11x + YOLO11xseg + YOLO11xpos pipeline for local/offline inference."""
from .schema.visionseg import VisionSeg, VisionSegPolicy, VisionSegConfig
from .runtime.pipeline import VisionSegPipeline
