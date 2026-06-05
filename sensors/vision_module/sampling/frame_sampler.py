from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import time

@dataclass
class FrameSampler:
    """نمونه‌بردار پیشرفته با قابلیت‌های بیشتر"""
    
    fps_infer: float
    adaptive: bool = False  # نمونه‌برداری تطبیقی
    min_fps: float = 1.0    # حداقل FPS
    max_fps: float = 30.0   # حداکثر FPS
    
    def __post_init__(self) -> None:
        self._last_sample_ms: Optional[int] = None
        self._interval_ms = self._calculate_interval()
        self._stats = {
            'total_frames': 0,
            'sampled_frames': 0,
            'skipped_frames': 0,
            'sampling_rate': 0.0
        }
    
    def _calculate_interval(self) -> int:
        """محاسبه فاصله زمانی"""
        fps = max(min(self.fps_infer, self.max_fps), self.min_fps)
        return int(1000.0 / max(fps, 0.001))  # جلوگیری از تقسیم بر صفر
    
    def should_sample(self, t_ms: int, frame_data: Optional[dict] = None) -> Tuple[bool, float]:
        """
        تصمیم‌گیری برای نمونه‌برداری
        
        Returns:
            (should_sample, actual_interval)
        """
        self._stats['total_frames'] += 1
        
        # نمونه‌برداری تطبیقی
        if self.adaptive and frame_data:
            self._adjust_for_motion(frame_data)
        
        # منطق اصلی نمونه‌برداری
        if self._last_sample_ms is None:
            self._last_sample_ms = t_ms
            self._stats['sampled_frames'] += 1
            return True, 0
        
        elapsed = t_ms - self._last_sample_ms
        
        if elapsed >= self._interval_ms:
            actual_interval = elapsed
            self._last_sample_ms = t_ms
            self._stats['sampled_frames'] += 1
            
            # محاسبه نرخ نمونه‌برداری واقعی
            self._stats['sampling_rate'] = (
                self._stats['sampled_frames'] / self._stats['total_frames']
            )
            
            return True, actual_interval
        else:
            self._stats['skipped_frames'] += 1
            return False, elapsed
    
    def _adjust_for_motion(self, frame_data: dict) -> None:
        """تنظیم تطبیقی بر اساس حرکت در صحنه"""
        motion_level = frame_data.get('motion_score', 0.0)
        
        # اگر حرکت زیادی باشد، نرخ نمونه‌برداری را افزایش بده
        if motion_level > 0.7:
            self.fps_infer = min(self.fps_infer * 1.5, self.max_fps)
        elif motion_level < 0.3:
            self.fps_infer = max(self.fps_infer * 0.8, self.min_fps)
        
        self._interval_ms = self._calculate_interval()
    
    def reset(self) -> None:
        """بازنشانی نمونه‌بردار"""
        self._last_sample_ms = None
        self._stats = {k: 0 for k in self._stats}
    
    def get_stats(self) -> dict:
        """دریافت آمار نمونه‌برداری"""
        return self._stats.copy()
    
    def get_current_fps(self) -> float:
        """دریافت FPS فعلی"""
        return 1000.0 / self._interval_ms if self._interval_ms > 0 else 0.0

