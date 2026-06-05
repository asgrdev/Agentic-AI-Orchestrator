from __future__ import annotations
from .. import pipeline as _src_mod
globals().update({k: v for k, v in vars(_src_mod).items() if not k.startswith("__")})

# Extracted methods for VisionSegPipeline from pipeline.py

def _initialize_components(self, runtime: RuntimeConfig, 
                          model_paths: ModelPaths, 
                          device: str,
                          embedding_store_dir: str,
                          snapshot_store_dir: str) -> None:
    """مقداردهی اولیه تمام کامپوننت‌ها"""
    
    # Sampling
    self.sampler = FrameSampler(fps_infer=runtime.fps_infer)
    logger.info(f"📊 Sampler با {runtime.fps_infer} FPS راه‌اندازی شد")
    
    # Quality estimation
    self.quality_estimator = FrameQualityEstimator()
    logger.info("📈 Quality Estimator راه‌اندازی شد")
    
    # Feature aggregation
    self.feature_aggregator = FeatureAggregator()
    logger.info("🔍 Feature Aggregator راه‌اندازی شد")
    
    # Detector
    detector_config = YoloConfig(
        model_path=model_paths.detector_path,
        device=device,
        conf_threshold=runtime.detection_conf_threshold,
        iou_threshold=runtime.detection_iou_threshold,
        max_det=runtime.max_detections_per_frame
    )
    self.detector = YoloDetector(detector_config)
    logger.info(f"🎯 Detector با مدل {model_paths.detector_path} راه‌اندازی شد")
    
    # Segmenter (optional)
    self.segmenter = None
    if runtime.enable_segmentation and model_paths.segmenter_path:
        segmenter_config = YoloConfig(
            model_path=model_paths.segmenter_path,
            device=device,
            conf_threshold=runtime.detection_conf_threshold,
            iou_threshold=runtime.detection_iou_threshold,
            max_det=runtime.max_detections_per_frame
        )
        self.segmenter = YoloSegmenter(segmenter_config)
        logger.info(f"🎭 Segmenter با مدل {model_paths.segmenter_path} راه‌اندازی شد")
    
    # Pose estimator (optional)
    self.pose = None
    if runtime.enable_pose and model_paths.pose_path:
        pose_config = YoloConfig(
            model_path=model_paths.pose_path,
            device=device,
            conf_threshold=runtime.detection_conf_threshold,
            iou_threshold=runtime.detection_iou_threshold,
            max_det=runtime.max_detections_per_frame
        )
        self.pose = YoloPose(pose_config)
        logger.info(f"💃 Pose estimator با مدل {model_paths.pose_path} راه‌اندازی شد")
    
    # Tracker (optional)
    self.tracker = None
    if runtime.tracker_enabled:
        self.tracker = SimpleIouTracker(
            iou_match_threshold=runtime.tracker_iou_match_threshold,
            track_ttl_sec=runtime.track_ttl_sec
        )
        logger.info(f"🔄 Tracker با TTL {runtime.track_ttl_sec} ثانیه راه‌اندازی شد")
    
    # Policy
    self.policy = VisionPolicy(
        enable_segmentation=(runtime.enable_segmentation and self.segmenter is not None),
        enable_pose=(runtime.enable_pose and self.pose is not None),
        seg_requires_person=runtime.segmentation_requires_person,
        pose_requires_person=runtime.pose_requires_person
    )
    logger.info("⚖️ Policy manager راه‌اندازی شد")
    
    # Scene classifier (optional)
    self.scene_classifier = None
    if runtime.scene_classifier_enabled:
        self.scene_classifier = SceneClassifier()
        logger.info("🏞️ Scene classifier راه‌اندازی شد")
    
    # Projector for embeddings (optional)
    self.projector = None
    if runtime.embedding_enabled:
        projector_config = ProjectorConfig(
            embedding_dim=runtime.embedding_dim,
            device=device
        )
        self.projector = VisionProjector(projector_config)
        logger.info(f"📐 Projector با بعد {runtime.embedding_dim} راه‌اندازی شد")
    
    # Stores
    self.embedding_store = EmbeddingStore(embedding_store_dir)
    self.snapshot_store = SnapshotStore(
        snapshot_store_dir,
        jpeg_quality=runtime.snapshot_jpeg_quality
    )
    logger.info(f"💾 Storage با موفقیت راه‌اندازی شد")

def process_frame(self, frame: VideoFrame) -> Optional[VisionSeg]:
    """
    پردازش یک فریم ویدیویی
    
    Args:
        frame: فریم ویدیویی ورودی
        
    Returns:
        VisionSeg object در صورت کامل شدن یک پنجره، در غیر این صورت None
    """
    if self.state in [PipelineState.ERROR, PipelineState.CLOSED]:
        logger.warning(f"⚠️ تلاش برای پردازش فریم در وضعیت {self.state}")
        return None
    
    self.state = PipelineState.PROCESSING
    start_time = time.time()
    
    try:
        # بررسی اعتبار فریم ورودی
        self._validate_frame(frame)
        
        # مدیریت پنجره زمانی
        if self._window_t0_ms is None:
            self._reset_window(next_t0_ms=frame.t_ms)
            self._last_snapshot_ms = frame.t_ms
        
        # تصمیم‌گیری برای نمونه‌برداری
        should_sample = self.sampler.should_sample(frame.t_ms)
        
        if not should_sample:
            self._previous_frame_bgr = frame.bgr
            self.metrics.frames_skipped += 1
            result = self._maybe_emit(frame, force=False)
            self.metrics.update_processing_time((time.time() - start_time) * 1000)
            self.state = PipelineState.READY
            return result
        
        # تخمین کیفیت فریم
        quality = self._estimate_frame_quality(frame)
        
        # تشخیص اشیاء
        detections = self._run_detection(frame)
        self._window_frames_used += 1
        self._last_frame_shape = (frame.bgr.shape[0], frame.bgr.shape[1])
        
        # ردیابی (اگر فعال باشد)
        tracked_detections = self._apply_tracking(detections, frame.t_ms)
        
        # آنالیز جمعیت
        self._analyze_population(detections)
        
        # تصمیم‌گیری policy
        gating = self._apply_policy(len(tracked_detections))

        seg_preview: List[Dict[str, Any]] = []
        pose_preview: List[Dict[str, Any]] = []
        
        # اجرای segmentation (اگر نیاز باشد)
        if gating.run_segmentation and self.segmenter is not None:
            seg_preview = self._run_segmentation(frame)
        
        # اجرای pose estimation (اگر نیاز باشد)
        if gating.run_pose and self.pose is not None:
            pose_preview = self._run_pose_estimation(frame)

        self._update_preview_annotations(
            tracked_detections=tracked_detections,
            seg_preview=seg_preview,
            pose_preview=pose_preview,
            frame_t_ms=int(frame.t_ms),
        )
        
        # ثبت snapshot (در صورت نیاز)
        self._capture_snapshot_if_needed(frame)
        
        # ذخیره فریم قبلی
        self._previous_frame_bgr = frame.bgr
        
        # بررسی برای انتشار پنجره
        result = self._maybe_emit(frame, force=False)
        
        self.metrics.update_processing_time((time.time() - start_time) * 1000)
        self.state = PipelineState.READY
        return result
        
    except Exception as e:
        self.state = PipelineState.ERROR
        error_msg = f"خطا در پردازش فریم {frame.frame_id}: {e}"
        self.metrics.record_error(error_msg)
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # تلاش برای بازیابی
        self._recover_from_error()
        
        return None

def flush(self, source_id: str, t_ms: int) -> Optional[VisionSeg]:
    """
    تخلیه داده‌های باقی‌مانده در پایپ‌لاین
    
    Args:
        source_id: شناسه منبع
        t_ms: زمان فعلی به میلی‌ثانیه
        
    Returns:
        VisionSeg object در صورت وجود داده، در غیر این صورت None
    """
    if self._window_t0_ms is None or self._window_frames_used == 0:
        logger.info("🚮 هیچ داده‌ای برای تخلیه وجود ندارد")
        return None
    
    logger.info(f"🚀 تخلیه پایپ‌لاین با {self._window_frames_used} فریم پردازش شده")
    
    dummy_bgr = (self._previous_frame_bgr if self._previous_frame_bgr is not None 
                else np.zeros((100, 100, 3), dtype=np.uint8))
    
    dummy_frame = VideoFrame(
        frame_id=-1,
        t_ms=t_ms,
        bgr=dummy_bgr,
        source_id=source_id
    )
    
    return self._maybe_emit(dummy_frame, force=True)

def get_status(self) -> Dict[str, Any]:
    """دریافت وضعیت فعلی پایپ‌لاین"""
    status = {
        "state": self.state.value,
        "metrics": {
            "frames_processed": self.metrics.frames_processed,
            "frames_skipped": self.metrics.frames_skipped,
            "windows_emitted": self.metrics.windows_emitted,
            "errors": self.metrics.errors,
            "avg_processing_time_ms": round(self.metrics.processing_time_avg, 2),
            "last_error": self.metrics.last_error,
            "last_error_time": self.metrics.last_error_time,
        },
        "window_info": {
            "current_window_id": self._current_window_id,
            "window_t0_ms": self._window_t0_ms,
            "frames_used": self._window_frames_used,
            "person_counts": self._person_counts,
            "object_histogram": dict(self._object_hist),
        },
        "components": {
            "detector": self.detector is not None,
            "segmenter": self.segmenter is not None,
            "pose": self.pose is not None,
            "tracker": self.tracker is not None,
            "scene_classifier": self.scene_classifier is not None,
            "projector": self.projector is not None,
        },
        "dataset_resources": {
            "count": len(self.dataset_resources),
            "keys": sorted(self.dataset_resources.keys()),
        },
    }
    return status

def close(self) -> None:
    """بستن پایپ‌لاین و آزادسازی منابع"""
    logger.info("🔚 در حال بستن پایپ‌لاین...")
    self.state = PipelineState.CLOSED
    
    # تخلیه هر پنجره باقی‌مانده
    if self._window_frames_used > 0:
        logger.info("تخلیه پنجره باقی‌مانده...")
    
    # پاکسازی منابع
    self._cleanup_resources()
    
    logger.info("✅ پایپ‌لاین با موفقیت بسته شد")

def _validate_frame(self, frame: VideoFrame) -> None:
    """اعتبارسنجی فریم ورودی"""
    if frame.bgr is None:
        raise ValueError("فریم ورودی None است")
    
    if not isinstance(frame.bgr, np.ndarray):
        raise TypeError(f"فریم باید numpy.ndarray باشد، نه {type(frame.bgr)}")
    
    if len(frame.bgr.shape) != 3 or frame.bgr.shape[2] != 3:
        raise ValueError(f"شکل فریم نامعتبر: {frame.bgr.shape}. انتظار (H, W, 3)")
    
    if frame.bgr.size == 0:
        raise ValueError("فریم ورودی خالی است")

def _estimate_frame_quality(self, frame: VideoFrame) -> FrameQuality:
    """تخمین کیفیت فریم"""
    try:
        quality = self.quality_estimator.estimate(
            frame.bgr, 
            self._previous_frame_bgr
        )
        
        # جمع‌آوری آمار کیفیت
        for k in self._quality_accum:
            self._quality_accum[k] += float(quality.get(k, 0.0))
        self._quality_n += 1
        
        return quality
        
    except Exception as e:
        logger.warning(f"خطا در تخمین کیفیت فریم: {e}")
        # مقادیر پیش‌فرض
        return FrameQuality(
            mean_brightness=0.5,
            blur_score=0.0,
            motion_score=0.0,
            occlusion_score=0.0
        )

def _run_detection(self, frame: VideoFrame) -> List[Tuple[str, float, np.ndarray]]:
    """اجرای تشخیص اشیاء"""
    try:
        return self.detector.infer(frame.bgr)
    except Exception as e:
        logger.error(f"خطا در تشخیص اشیاء: {e}")
        return []

