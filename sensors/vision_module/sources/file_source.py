from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List
import time
import os
import sys
import threading
from queue import Queue
import numpy as np
from .base import VideoSource, VideoFrame
from ..cv2_compat import cv2, CV2_AVAILABLE, CV2_IMPORT_ERROR


@dataclass
class FileVideoSource(VideoSource):
    """منبع ویدیو از فایل با خطایابی پیشرفته"""
    file_path: str
    source_id: str = "file"
    realtime: bool = False
    loop: bool = False  # افزودن قابلیت لوپ
    max_retries: int = 3  # تعداد تلاش‌های مجدد برای باز کردن فایل
    
    def __post_init__(self) -> None:
        if not CV2_AVAILABLE:
            raise ImportError(
                f"opencv-python is required but could not be imported. "
                f"Error: {CV2_IMPORT_ERROR}\n"
                f"Install with: pip install opencv-python"
            )
        
        # بررسی وجود فایل
        self._validate_file()
        
        # باز کردن فایل ویدیویی با تلاش مجدد
        self._cap = self._open_video_with_retry()
        
        # استخراج metadata
        self._extract_video_metadata()
        
        # متغیرهای زمان‌بندی
        self._started = time.monotonic()
        self._frame_id = 0
        self._total_frames_read = 0
        
        # برای realtime
        self._last_frame_time = 0
        self._frame_interval = 1.0 / max(self._fps, 1.0)  # جلوگیری از تقسیم بر صفر
        
        print(f"✅ ویدیو باز شد: {os.path.basename(self.file_path)}")
        print(f"   اندازه: {self._width}x{self._height}, FPS: {self._fps:.2f}, "
              f"فریم‌ها: {self._frame_count}, کدک: {self._fourcc_str}")

    def _validate_file(self) -> None:
        """بررسی وجود و خوانایی فایل"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"فایل ویدیو پیدا نشد: {self.file_path}\n"
                f"مسیر کامل: {os.path.abspath(self.file_path)}"
            )
        
        if not os.path.isfile(self.file_path):
            raise ValueError(f"مسیر وارد شده یک فایل نیست: {self.file_path}")
        
        # بررسی پسوند فایل
        valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v'}
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in valid_extensions:
            print(f"⚠️ هشدار: پسوند فایل {ext} ممکن است پشتیبانی نشود. "
                  f"پسوندهای توصیه شده: {', '.join(valid_extensions)}")
        
        # بررسی سایز فایل
        file_size = os.path.getsize(self.file_path) / (1024 * 1024)  # MB
        if file_size == 0:
            raise ValueError(f"فایل ویدیو خالی است (سایز: 0 بایت): {self.file_path}")
        print(f"   سایز فایل: {file_size:.2f} MB")

    def _open_video_with_retry(self) -> cv2.VideoCapture:
        """باز کردن فایل ویدیویی با تلاش‌های مجدد"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # استفاده از CAP_ANY برای اتوماتیک تشخیص backend
                cap = cv2.VideoCapture(self.file_path, cv2.CAP_ANY)
                
                # امتحان backendهای مختلف در صورت شکست
                if not cap.isOpened():
                    backends = [
                        cv2.CAP_FFMPEG,
                        cv2.CAP_IMAGES if hasattr(cv2, 'CAP_IMAGES') else None,
                        cv2.CAP_MSMF if hasattr(cv2, 'CAP_MSMF') else None,
                    ]
                    
                    for backend in backends:
                        if backend is not None:
                            cap = cv2.VideoCapture(self.file_path, backend)
                            if cap.isOpened():
                                print(f"   استفاده از backend: {backend}")
                                break
                
                if cap.isOpened():
                    if attempt > 0:
                        print(f"   موفق در تلاش {attempt + 1}")
                    return cap
                    
            except Exception as e:
                last_error = e
            
            if attempt < self.max_retries - 1:
                print(f"   تلاش مجدد {attempt + 1}/{self.max_retries}...")
                time.sleep(0.1)
        
        # اگر به اینجا رسیدیم، همه تلاش‌ها شکست خورده
        error_msg = f"ناتوان در باز کردن فایل ویدیو: {self.file_path}"
        if last_error:
            error_msg += f"\nخطا: {last_error}"
        
        # پیشنهاد راه‌حل
        error_msg += (
            "\n\nراه‌حل‌های احتمالی:\n"
            "1. مطمئن شوید فایل وجود دارد و قابل خواندن است\n"
            "2. codecهای لازم را نصب کنید:\n"
            "   - Windows: K-Lite Codec Pack\n"
            "   - Ubuntu: sudo apt install ubuntu-restricted-extras\n"
            "3. فایل را با ffmpeg بازنویسی کنید:\n"
            "   ffmpeg -i input.mp4 -c:v libx264 -c:a aac output.mp4\n"
            "4. از مسیر کامل (full path) استفاده کنید\n"
            "5. از کاراکترهای غیر ASCII در مسیر پرهیز کنید"
        )
        
        raise RuntimeError(error_msg)

    def _extract_video_metadata(self) -> None:
        """استخراج metadata ویدیو"""
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self._fps <= 0:
            self._fps = 25.0  # مقدار پیش‌فرض
            print(f"⚠️ FPS تشخیص داده نشد، استفاده از مقدار پیش‌فرض: {self._fps}")
        
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # تشخیص codec
        fourcc_int = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self._fourcc_str = self._decode_fourcc(fourcc_int)
        
        # تنظیمات اضافی برای بهبود performance
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # کاهش بافر برای realtime

    @staticmethod
    def _decode_fourcc(fourcc: int) -> str:
        """تبدیل FOURCC integer به string"""
        if fourcc == 0:
            return "Unknown"
        return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def read(self, timeout: Optional[float] = None) -> Optional[VideoFrame]:
        """خواندن فریم بعدی با مدیریت خطا"""
        _ = timeout
        try:
            ok, frame = self._cap.read()
            
            # اگر به پایان فایل رسیدیم
            if not ok:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._cap.read()
                    if ok:
                        print("🔁 ویدیو از ابتدا پخش مجدد شد")
                    else:
                        return None
                else:
                    return None
            
            # مدیریت realtime
            if self.realtime:
                current_time = time.monotonic()
                expected_time = self._started + (self._frame_id * self._frame_interval)
                
                if current_time < expected_time:
                    sleep_time = expected_time - current_time
                    if sleep_time > 0.001:  # فقط اگر قابل توجه است
                        time.sleep(sleep_time)
                elif current_time > expected_time + self._frame_interval * 2:
                    # اگر خیلی عقب افتادیم، reset کنیم
                    self._started = current_time - (self._frame_id * self._frame_interval)
            
            self._frame_id += 1
            self._total_frames_read += 1
            
            return VideoFrame(
                frame_id=self._frame_id,
                t_ms=int(time.time() * 1000),
                bgr=frame,
                source_id=self.source_id,
                metadata={
                    'fps': self._fps,
                    'width': self._width,
                    'height': self._height,
                    'position': self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                }
            )
            
        except Exception as e:
            print(f"⚠️ خطا در خواندن فریم: {e}")
            # در صورت خطا، یک فریم سیاه برگردانید یا None
            return None

    def get_progress(self) -> Tuple[float, str]:
        """درصد پیشرفت پخش ویدیو"""
        if self._frame_count > 0:
            current_frame = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = (current_frame / self._frame_count) * 100
            return progress, f"{progress:.1f}%"
        return 0.0, "نامعلوم"

    def seek(self, position_sec: float) -> bool:
        """پرش به زمان خاص در ویدیو (ثانیه)"""
        try:
            # تبدیل ثانیه به میلی‌ثانیه
            position_ms = position_sec * 1000
            success = self._cap.set(cv2.CAP_PROP_POS_MSEC, position_ms)
            if success:
                self._frame_id = int(position_sec * self._fps)
                print(f"↪️ پرش به {position_sec:.2f} ثانیه")
            return success
        except Exception as e:
            print(f"⚠️ خطا در seek: {e}")
            return False

    def close(self) -> None:
        """بستن منبع ویدیو با مدیریت خطا"""
        try:
            if hasattr(self, '_cap') and self._cap is not None:
                self._cap.release()
                print(f"✅ ویدیو بسته شد: {self.source_id}")
        except Exception as e:
            print(f"⚠️ خطا در بستن ویدیو: {e}")
        finally:
            self._cap = None

    # Context manager support
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # Iterator support
    def __iter__(self):
        return self
    
    def __next__(self) -> VideoFrame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame


@dataclass
class ThreadedFileVideoSource(FileVideoSource):
    """نسخه thread-safe برای خواندن ویدیو در برنامه‌های real-time"""
    
    def __post_init__(self):
        super().__post_init__()
        self._queue = Queue(maxsize=30)  # بافر 30 فریم
        self._thread = None
        self._running = False
        
    def start(self):
        """شروع thread برای خواندن ویدیو"""
        self._running = True
        self._thread = threading.Thread(target=self._read_frames, daemon=True)
        self._thread.start()
        print("🧵 Thread خواندن ویدیو شروع شد")
    
    def _read_frames(self):
        """تابع thread برای خواندن فریم‌ها"""
        while self._running:
            frame = super().read()
            if frame is None:
                self._running = False
                break
            
            # اگر صف پر است، قدیمی‌ترین را دور بریز
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except:
                    pass
            
            self._queue.put(frame)
    
    def read(self, timeout: Optional[float] = 1.0) -> Optional[VideoFrame]:
        """خواندن از صف thread-safe"""
        if not self._running and self._thread is None:
            self.start()
        
        try:
            return self._queue.get(timeout=timeout)
        except:
            return None
    
    def close(self):
        """بستن thread و منبع"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        super().close()


# تابع کمکی برای تشخیص اطلاعات ویدیو
def get_video_info(file_path: str) -> dict:
    """دریافت اطلاعات ویدیو بدون باز کردن کامل"""
    if not CV2_AVAILABLE:
        return {"error": "OpenCV not installed"}
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return {"error": "Cannot open video file"}
    
    info = {
        'fps': float(cap.get(cv2.CAP_PROP_FPS)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(float(cap.get(cv2.CAP_PROP_FPS)), 1.0),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    
    cap.release()
    return info
