from __future__ import annotations
from .. import pipeline as _src_mod
globals().update({k: v for k, v in vars(_src_mod).items() if not k.startswith("__")})

# Extracted methods for VisionSegPipeline from pipeline.py

def _apply_tracking(self, detections: List[Tuple[str, float, np.ndarray]], 
                   t_ms: int) -> List[Tuple[Optional[int], str, float, np.ndarray]]:
    """اعمال ردیابی روی تشخیص‌ها"""
    if self.tracker is not None:
        try:
            return self.tracker.update(detections, t_ms)
        except Exception as e:
            logger.warning(f"خطا در ردیابی: {e}")
    
    # اگر ردیابی فعال نیست یا خطا دارد
    return [(None, n, c, b) for n, c, b in detections]

def _analyze_population(self, detections: List[Tuple[str, float, np.ndarray]]) -> None:
    """تحلیل جمعیت و اشیاء"""
    person_count = sum(1 for n, _, _ in detections if n == "person")
    self._person_counts.append(person_count)
    
    for n, _, _ in detections:
        self._object_hist[n] = self._object_hist.get(n, 0) + 1

def _apply_policy(self, person_present: bool) -> Any:
    """اعمال policy برای تصمیم‌گیری"""
    try:
        return self.policy.decide(person_present=person_present)
    except Exception as e:
        logger.warning(f"خطا در اعمال policy: {e}")
        # مقادیر پیش‌فرض
        class DefaultGating:
            run_segmentation = False
            run_pose = False
        return DefaultGating()

def _run_segmentation(self, frame: VideoFrame) -> List[Dict[str, Any]]:
    """اجرای segmentation"""
    preview_rows: List[Dict[str, Any]] = []
    try:
        result = self.segmenter.infer(frame.bgr)
        
        if (
            hasattr(result, "masks")
            and result.masks is not None
            and hasattr(result, "boxes")
            and result.boxes is not None
        ):
            masks = result.masks.data.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            names = result.names
            
            n = int(min(masks.shape[0], len(confs), len(classes), len(boxes_xyxy)))
            for i in range(n):
                mask_data = masks[i]
                confidence = float(confs[i])
                class_name = names.get(int(classes[i]), f"cls_{classes[i]}")
                area_ratio = float(np.mean(mask_data > 0.5))
                bb = boxes_xyxy[i]
                bbox_xyxy = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
                
                self._masks.append(
                    SegmentationMask(
                        frame_id=frame.frame_id,
                        track_id=None,
                        class_name=class_name,
                        confidence=confidence,
                        area_ratio=area_ratio
                    )
                )
                preview_rows.append(
                    {
                        "label": str(class_name),
                        "confidence": float(confidence),
                        "bbox_xyxy": bbox_xyxy,
                        "seg_area_ratio": float(area_ratio),
                        "_preview_seg": True,
                    }
                )
                
    except Exception as e:
        logger.warning(f"خطا در segmentation: {e}")
    return preview_rows

def _run_pose_estimation(self, frame: VideoFrame) -> List[Dict[str, Any]]:
    """اجرای تخمین pose"""
    preview_rows: List[Dict[str, Any]] = []
    try:
        result = self.pose.infer(frame.bgr)
        
        if hasattr(result, "keypoints") and result.keypoints is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = (result.keypoints.conf.cpu().numpy() 
                            if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None 
                            else np.ones(keypoints_xy.shape[:2], dtype="float32"))
            
            box_confs = (result.boxes.conf.cpu().numpy() 
                       if hasattr(result, "boxes") and result.boxes is not None 
                       else np.ones((keypoints_xy.shape[0],), dtype="float32"))
            box_xyxy = (
                result.boxes.xyxy.cpu().numpy()
                if hasattr(result, "boxes") and result.boxes is not None and hasattr(result.boxes, "xyxy")
                else np.zeros((keypoints_xy.shape[0], 4), dtype="float32")
            )
            
            for i in range(keypoints_xy.shape[0]):
                points = []
                for j in range(keypoints_xy.shape[1]):
                    x, y = float(keypoints_xy[i, j, 0]), float(keypoints_xy[i, j, 1])
                    conf = float(keypoints_conf[i, j]) if j < keypoints_conf.shape[1] else 1.0
                    points.append((x, y, conf))
                
                pose_quality = float(np.mean([p[2] for p in points]) if points else 0.0)
                
                self._poses.append(
                    PoseKeypoints(
                        frame_id=frame.frame_id,
                        track_id=None,
                        confidence=float(box_confs[i]),
                        keypoints=points,
                        pose_quality=pose_quality
                    )
                )
                bbox_xyxy = [float(box_xyxy[i, 0]), float(box_xyxy[i, 1]), float(box_xyxy[i, 2]), float(box_xyxy[i, 3])]
                preview_rows.append(
                    {
                        "label": "person",
                        "confidence": float(box_confs[i]),
                        "bbox_xyxy": bbox_xyxy,
                        "keypoints": list(points),
                        "pose_quality": float(pose_quality),
                        "_preview_pose": True,
                    }
                )
                
    except Exception as e:
        logger.warning(f"خطا در تخمین pose: {e}")
    return preview_rows

def _capture_snapshot_if_needed(self, frame: VideoFrame) -> None:
    """ثبت snapshot در صورت نیاز"""
    if (self.runtime.snapshot_every_sec > 0 and 
        self._last_snapshot_ms is not None and
        (frame.t_ms - self._last_snapshot_ms) >= int(self.runtime.snapshot_every_sec * 1000)):
        
        self._last_snapshot_ms = frame.t_ms
        
        try:
            snapshot_path = self.snapshot_store.save(
                self._current_window_id,
                frame.frame_id,
                frame.bgr
            )
            self._snapshots.append(snapshot_path)
            logger.debug(f"📸 Snapshot ثبت شد: {snapshot_path}")
        except Exception as e:
            logger.warning(f"خطا در ثبت snapshot: {e}")

def _maybe_emit(self, frame: VideoFrame, force: bool) -> Optional[VisionSeg]:
    """
    بررسی و انتشار پنجره در صورت کامل شدن
    
    Args:
        frame: فریم فعلی
        force: آیا پنجره را مجبور به انتشار کنیم؟
        
    Returns:
        VisionSeg object یا None
    """
    if self._window_t0_ms is None:
        return None
    
    elapsed_ms = frame.t_ms - self._window_t0_ms
    should_emit = force or (elapsed_ms >= int(self.runtime.window_duration_sec * 1000))
    
    if not should_emit:
        return None
    
    try:
        # ساخت و انتشار پنجره
        vision_seg = self._build(source_id=frame.source_id, t1_ms=frame.t_ms)
        self.metrics.windows_emitted += 1
        
        logger.info(f"📤 انتشار پنجره {self._current_window_id} "
                   f"با {self._window_frames_used} فریم")
        
        # بازنشانی برای پنجره بعدی
        self._reset_window(next_t0_ms=frame.t_ms)
        self._last_snapshot_ms = frame.t_ms
        
        return vision_seg
        
    except Exception as e:
        error_msg = f"خطا در ساخت VisionSeg: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        self.metrics.record_error(error_msg)
        
        # بازنشانی پنجره حتی در صورت خطا
        self._reset_window(next_t0_ms=frame.t_ms)
        return None

