from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import uuid
import logging
import time
import traceback
import numpy as np
from enum import Enum
import gc

from ..config import RuntimeConfig
from ..schema.visionseg import (
    VisionSeg, FrameQuality, Detection, SegmentationMask, PoseKeypoints,
    DetectorStats, DetectorResult, SegmentationResult, PoseResult, SceneResult,
    EmbeddingResult, VisionSegPolicy, VisionSegConfig
)
from ..sources.base import VideoFrame
from ..sampling.frame_sampler import FrameSampler
from ..processors.quality import FrameQualityEstimator
from ..processors.tracker import SimpleIouTracker
from ..processors.yolo_wrappers import YoloConfig, YoloDetector, YoloSegmenter, YoloPose
from ..processors.policy import VisionPolicy
from ..processors.features import FeatureAggregator
from ..processors.scene_classifier import SceneClassifier
from ..processors.projector import ProjectorConfig, VisionProjector
from ..sinks.embedding_store import EmbeddingStore
from ..sinks.snapshot_store import SnapshotStore

logger = logging.getLogger("vision_module")


class PipelineState(Enum):
    """وضعیت‌های مختلف pipeline"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class ModelPaths:
    """مسیرهای مدل‌های مورد نیاز"""
    detector_path: str
    segmenter_path: Optional[str] = None
    pose_path: Optional[str] = None
    
    def validate(self) -> List[str]:
        """اعتبارسنجی مسیرهای مدل"""
        errors = []
        import os
        
        if not os.path.exists(self.detector_path):
            errors.append(f"مسیر مدل detector یافت نشد: {self.detector_path}")
        
        if self.segmenter_path and not os.path.exists(self.segmenter_path):
            errors.append(f"مسیر مدل segmenter یافت نشد: {self.segmenter_path}")
        
        if self.pose_path and not os.path.exists(self.pose_path):
            errors.append(f"مسیر مدل pose یافت نشد: {self.pose_path}")
        
        return errors


@dataclass
class PipelineMetrics:
    """آمار و متریک‌های pipeline"""
    frames_processed: int = 0
    frames_skipped: int = 0
    windows_emitted: int = 0
    errors: int = 0
    processing_time_total: float = 0.0
    processing_time_avg: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    
    def update_processing_time(self, time_ms: float):
        """به‌روزرسانی زمان پردازش"""
        self.frames_processed += 1
        self.processing_time_total += time_ms
        self.processing_time_avg = self.processing_time_total / self.frames_processed
    
    def record_error(self, error: str):
        """ثبت خطا"""
        self.errors += 1
        self.last_error = error
        self.last_error_time = time.time()


class VisionSegPipeline:
    """پایپ‌لاین اصلی پردازش بینایی کامپیوتر"""
    
    def __init__(self, 
                 runtime: RuntimeConfig, 
                 model_paths: ModelPaths, 
                 device: str, 
                 embedding_store_dir: str, 
                 snapshot_store_dir: str) -> None:
        """
        مقداردهی اولیه پایپ‌لاین
        
        Args:
            runtime: تنظیمات زمان اجرا
            model_paths: مسیرهای مدل‌ها
            device: دستگاه پردازش (cpu/cuda)
            embedding_store_dir: دایرکتوری ذخیره embeddingها
            snapshot_store_dir: دایرکتوری ذخیره snapshotها
        """
        self.runtime = runtime
        self.device = device
        self.dataset_resources = dict(getattr(runtime, "dataset_resources", {}) or {})
        self.state = PipelineState.INITIALIZING
        self.metrics = PipelineMetrics()
        
        try:
            logger.info(f"📦 در حال راه‌اندازی VisionSegPipeline روی {device}...")
            
            # اعتبارسنجی مسیرهای مدل
            validation_errors = model_paths.validate()
            if validation_errors:
                raise ValueError(f"خطا در مسیرهای مدل:\n" + "\n".join(validation_errors))
            
            # مقداردهی کامپوننت‌ها
            self._initialize_components(runtime, model_paths, device, 
                                       embedding_store_dir, snapshot_store_dir)
            
            # تنظیمات پنجره
            self._reset_window(next_t0_ms=None)
            self._previous_frame_bgr: Optional[np.ndarray] = None
            self._last_frame_shape: Optional[Tuple[int, int]] = None
            self._last_snapshot_ms: Optional[int] = None
            self._latest_preview_annotations: List[Dict[str, Any]] = []
            
            self.state = PipelineState.READY
            logger.info("✅ VisionSegPipeline با موفقیت راه‌اندازی شد")
            
        except Exception as e:
            self.state = PipelineState.ERROR
            self.metrics.record_error(str(e))
            logger.error(f"❌ خطا در راه‌اندازی پایپ‌لاین: {e}")
            logger.error(traceback.format_exc())
            raise
    
    
    
    
    
    
    # ========== متدهای خصوصی ==========
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    # ========== Context Manager Support ==========
    
    


# تابع کمکی برای ایجاد pipeline
def create_pipeline(
    runtime_config_path: Optional[str] = None,
    model_paths: Optional[ModelPaths] = None,
    device: str = "cpu",
    embedding_store_dir: str = "./embeddings",
    snapshot_store_dir: str = "./snapshots"
) -> VisionSegPipeline:
    """
    ایجاد یک pipeline با تنظیمات پیش‌فرض
    
    Args:
        runtime_config_path: مسیر فایل تنظیمات زمان اجرا
        model_paths: مسیرهای مدل‌ها
        device: دستگاه پردازش
        embedding_store_dir: دایرکتوری ذخیره embeddingها
        snapshot_store_dir: دایرکتوری ذخیره snapshotها
    
    Returns:
        نمونه‌ای از VisionSegPipeline
    """
    from ..config import load_runtime_config
    
    # بارگذاری تنظیمات
    if runtime_config_path:
        runtime_config = load_runtime_config(runtime_config_path)
    else:
        # تنظیمات پیش‌فرض
        runtime_config = RuntimeConfig()
    
    # ایجاد pipeline
    return VisionSegPipeline(
        runtime=runtime_config,
        model_paths=model_paths or ModelPaths(detector_path="yolo11x.pt"),
        device=device,
        embedding_store_dir=embedding_store_dir,
        snapshot_store_dir=snapshot_store_dir
    )

# ---- extracted method wiring: VisionSegPipeline ----
from .pipeline_parts.lifecycle_and_detection import _initialize_components, process_frame, flush, get_status, close, _validate_frame, _estimate_frame_quality, _run_detection
from .pipeline_parts.tracking_policy_and_outputs import _apply_tracking, _analyze_population, _apply_policy, _run_segmentation, _run_pose_estimation, _capture_snapshot_if_needed, _maybe_emit
from .pipeline_parts.builders_annotations_and_recovery import _build, _reset_window, _update_preview_annotations, get_preview_annotations, _recover_from_error, _cleanup_resources, __enter__, __exit__
VisionSegPipeline._initialize_components = _initialize_components
VisionSegPipeline.process_frame = process_frame
VisionSegPipeline.flush = flush
VisionSegPipeline.get_status = get_status
VisionSegPipeline.close = close
VisionSegPipeline._validate_frame = _validate_frame
VisionSegPipeline._estimate_frame_quality = _estimate_frame_quality
VisionSegPipeline._run_detection = _run_detection
VisionSegPipeline._apply_tracking = _apply_tracking
VisionSegPipeline._analyze_population = _analyze_population
VisionSegPipeline._apply_policy = _apply_policy
VisionSegPipeline._run_segmentation = _run_segmentation
VisionSegPipeline._run_pose_estimation = _run_pose_estimation
VisionSegPipeline._capture_snapshot_if_needed = _capture_snapshot_if_needed
VisionSegPipeline._maybe_emit = _maybe_emit
VisionSegPipeline._build = _build
VisionSegPipeline._reset_window = _reset_window
VisionSegPipeline._update_preview_annotations = _update_preview_annotations
VisionSegPipeline.get_preview_annotations = get_preview_annotations
VisionSegPipeline._recover_from_error = _recover_from_error
VisionSegPipeline._cleanup_resources = _cleanup_resources
VisionSegPipeline.__enter__ = __enter__
VisionSegPipeline.__exit__ = __exit__
