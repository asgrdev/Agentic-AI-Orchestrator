from __future__ import annotations
from .. import pipeline as _src_mod
globals().update({k: v for k, v in vars(_src_mod).items() if not k.startswith("__")})

# Extracted methods for VisionSegPipeline from pipeline.py

def _build(self, source_id: str, t1_ms: int) -> VisionSeg:
    """ساخت object VisionSeg نهایی"""
    
    # محاسبه کیفیت متوسط
    quality_mean = {
        k: (self._quality_accum[k] / max(self._quality_n, 1))
        for k in self._quality_accum
    }
    quality = FrameQuality(**quality_mean)
    
    # آمار تشخیص
    person_mean = float(sum(self._person_counts) / max(len(self._person_counts), 1))
    person_max = int(max(self._person_counts) if self._person_counts else 0)
    stats = DetectorStats(
        person_count_mean=person_mean,
        person_count_max=person_max,
        object_histogram=dict(self._object_hist)
    )
    
    # اعتماد تشخیص
    det_conf_values = [d.confidence for d in self._detections]
    mean_det_conf = float(np.mean(det_conf_values)) if det_conf_values else 0.0
    quality_penalty = float(np.clip(1.0 - (max(0.0, 0.25 - quality.mean_brightness) * 1.5), 0.0, 1.0))
    detector_confidence = float(np.clip(mean_det_conf * quality_penalty, 0.0, 1.0))
    
    # نتایج تشخیص
    detector = DetectorResult(
        model_name="yolo11x",
        enabled=True,
        conf_threshold=self.runtime.detection_conf_threshold,
        iou_threshold=self.runtime.detection_iou_threshold,
        detections=self._detections,
        stats=stats,
        confidence=detector_confidence
    )
    
    # نتایج segmentation
    segmentation = None
    if self._masks:
        mask_confs = [m.confidence for m in self._masks]
        segmentation = SegmentationResult(
            model_name="yolo11xseg",
            enabled=True,
            masks=self._masks,
            confidence=float(np.clip(np.mean(mask_confs), 0.0, 1.0))
        )
    
    # نتایج pose
    pose = None
    if self._poses:
        pose_confs = [p.confidence for p in self._poses]
        pose = PoseResult(
            model_name="yolo11xpos",
            enabled=True,
            poses=self._poses,
            confidence=float(np.clip(np.mean(pose_confs), 0.0, 1.0))
        )
    
    # نتایج scene
    scene = None
    features = {}
    if self._last_frame_shape is not None and self._previous_frame_bgr is not None:
        # استخراج ویژگی‌ها
        sky_ratio = self.feature_aggregator.compute_sky_ratio(self._previous_frame_bgr)
        features = self.feature_aggregator.build_window_features(
            quality_mean,
            self._object_hist,
            self._window_frames_used,
            self._last_frame_shape
        )
        features["sky_ratio"] = float(sky_ratio)
        
        # طبقه‌بندی صحنه
        if self.scene_classifier is not None:
            try:
                l1, l2, conf = self.scene_classifier.infer(features)
                scene = SceneResult(
                    enabled=True,
                    label_level_1=l1,
                    label_level_2=l2,
                    confidence=float(conf),
                    features={k: float(v) for k, v in features.items() 
                             if isinstance(v, (int, float))}
                )
            except Exception as e:
                logger.warning(f"خطا در طبقه‌بندی صحنه: {e}")
    
    # نتایج embedding
    embedding = None
    if self.projector is not None and features:
        try:
            z = self.projector.infer(features)
            vec_path = self.embedding_store.save(self._current_window_id, z)
            embedding = EmbeddingResult(
                enabled=True,
                model_name=self.projector.cfg.model_name,
                dimension=int(z.shape[0]),
                vector_path=vec_path,
                l2_norm=float(np.linalg.norm(z) + 1e-12),
                confidence=float(np.clip(0.5 + detector_confidence * 0.5, 0.0, 1.0))
            )
        except Exception as e:
            logger.warning(f"خطا در تولید embedding: {e}")
    
    # تنظیمات policy
    policy = VisionSegPolicy(
        window_duration_sec=float(self.runtime.window_duration_sec),
        fps_infer=float(self.runtime.fps_infer),
        snapshot_every_sec=float(self.runtime.snapshot_every_sec),
        detection_conf_threshold=float(self.runtime.detection_conf_threshold),
        detection_iou_threshold=float(self.runtime.detection_iou_threshold),
        tracker_enabled=bool(self.runtime.tracker_enabled),
        track_ttl_sec=float(self.runtime.track_ttl_sec),
        enable_segmentation=bool(self.runtime.enable_segmentation),
        enable_pose=bool(self.runtime.enable_pose)
    )
    
    # ساخت object نهایی
    return VisionSeg(
        seg_id=self._current_window_id,
        source_id=source_id,
        t0_ms=int(self._window_t0_ms or t1_ms),
        t1_ms=int(t1_ms),
        fps_infer=float(self.runtime.fps_infer),
        frames_used=int(self._window_frames_used),
        quality=quality,
        detector=detector,
        segmentation=segmentation,
        pose=pose,
        scene=scene,
        embedding=embedding,
        snapshots=list(self._snapshots)
    )

def _reset_window(self, next_t0_ms: Optional[int]) -> None:
    """بازنشانی متغیرهای پنجره جاری"""
    self._current_window_id = str(uuid.uuid4())
    self._window_t0_ms = next_t0_ms
    self._window_frames_used = 0
    
    # متغیرهای آماری
    self._object_hist: Dict[str, int] = {}
    self._person_counts: List[int] = []
    self._quality_accum = {
        "mean_brightness": 0.0,
        "blur_score": 0.0,
        "motion_score": 0.0,
        "occlusion_score": 0.0
    }
    self._quality_n = 0
    
    # داده‌های پردازش شده
    self._detections: List[Detection] = []
    self._masks: List[SegmentationMask] = []
    self._poses: List[PoseKeypoints] = []
    self._snapshots: List[str] = []
    
    logger.debug(f"🔄 پنجره بازنشانی شد: {self._current_window_id}")

def _update_preview_annotations(
    self,
    tracked_detections: List[Tuple[Optional[int], str, float, np.ndarray]],
    seg_preview: List[Dict[str, Any]],
    pose_preview: List[Dict[str, Any]],
    frame_t_ms: int,
) -> None:
    rows: List[Dict[str, Any]] = []
    for tid, name, conf, bbox in tracked_detections:
        try:
            bb = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except Exception:
            continue
        d = Detection(
            frame_id=-1,
            class_name=str(name),
            confidence=float(conf),
            bbox_xyxy=(bb[0], bb[1], bb[2], bb[3]),
            track_id=(None if tid is None else int(tid)),
        )
        self._detections.append(d)
        rows.append(
            {
                "label": str(name),
                "confidence": float(conf),
                "bbox_xyxy": bb,
                "track_id": (None if tid is None else int(tid)),
                "_t_ms": int(frame_t_ms),
            }
        )

    for row in seg_preview or []:
        if not isinstance(row, dict):
            continue
        rr = dict(row)
        rr["_t_ms"] = int(frame_t_ms)
        rows.append(rr)

    for row in pose_preview or []:
        if not isinstance(row, dict):
            continue
        rr = dict(row)
        rr["_t_ms"] = int(frame_t_ms)
        rows.append(rr)

    self._latest_preview_annotations = rows

def get_preview_annotations(self, max_age_ms: int = 2000) -> List[Dict[str, Any]]:
    now_ms = int(time.time() * 1000)
    out: List[Dict[str, Any]] = []
    for row in self._latest_preview_annotations or []:
        if not isinstance(row, dict):
            continue
        t_ms = int(row.get("_t_ms", now_ms))
        if (now_ms - t_ms) > max(100, int(max_age_ms)):
            continue
        clean = dict(row)
        clean.pop("_t_ms", None)
        out.append(clean)
    return out

def _recover_from_error(self) -> None:
    """بازیابی از خطا"""
    logger.info("🔄 تلاش برای بازیابی از خطا...")
    
    # بازنشانی پنجره جاری
    if self._window_frames_used > 0:
        logger.warning(f"از دست دادن {self._window_frames_used} فریم پردازش شده")
        self._reset_window(next_t0_ms=None)
    
    # پاکسازی حافظه
    gc.collect()
    
    # تلاش برای بازگرداندن به حالت آماده
    self.state = PipelineState.READY

def _cleanup_resources(self) -> None:
    """پاکسازی منابع"""
    # پاکسازی مدل‌ها
    if hasattr(self.detector, 'cleanup'):
        self.detector.cleanup()
    
    if self.segmenter and hasattr(self.segmenter, 'cleanup'):
        self.segmenter.cleanup()
    
    if self.pose and hasattr(self.pose, 'cleanup'):
        self.pose.cleanup()
    
    # پاکسازی حافظه
    self._previous_frame_bgr = None
    self._detections.clear()
    self._masks.clear()
    self._poses.clear()
    self._snapshots.clear()
    self._latest_preview_annotations.clear()
    
    gc.collect()

def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
    if exc_type is not None:
        logger.error(f"خطا در context manager: {exc_val}")
        return False
    return True

