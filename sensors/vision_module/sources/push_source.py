from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any
import queue
import time
import threading
import numpy as np
from collections import deque
import warnings
from .base import VideoSource, VideoFrame

@dataclass
class PushFrameSource(VideoSource):
    """منبع ویدیوی push-based برای دریافت فریم‌ها از منابع خارجی"""
    source_id: str = "push0"
    max_queue: int = 200
    drop_oldest: bool = True  # اگر صف پر است، قدیمی‌ترین را حذف کن
    auto_frame_id: bool = True  # تولید خودکار frame_id
    stats_enabled: bool = True  # جمع‌آوری آمار
    
    def __post_init__(self) -> None:
        self._queue: queue.Queue[VideoFrame] = queue.Queue(maxsize=self.max_queue)
        self._frame_id = 0
        self._lock = threading.RLock()  # برای thread-safe operations
        self._closed = False
        self._stats = {
            'frames_pushed': 0,
            'frames_popped': 0,
            'queue_overflows': 0,
            'frames_dropped': 0,
            'push_errors': 0,
            'last_push_time': 0,
            'last_read_time': 0,
            'queue_size_history': deque(maxlen=100),
        }
        self._consumers = []  # برای callbackها
        self._frame_callbacks = []  # توابع فراخوانی برای هر فریم جدید
        
        # برای monitoring
        self._monitor_thread = None
        self._monitoring = False
        self._monitor_interval = 5.0  # ثانیه
        
        print(f"✅ PushFrameSource ایجاد شد (max_queue: {self.max_queue})")

    def push(self, bgr: np.ndarray, 
             t_ms: Optional[int] = None,
             frame_id: Optional[int] = None,
             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        ارسال یک فریم جدید به صف
        
        Args:
            bgr: فریم تصویر در فرمت BGR
            t_ms: زمان برحسب میلی‌ثانیه (اگر None باشد از زمان سیستم استفاده می‌شود)
            frame_id: شناسه فریم (اگر None باشد به صورت خودکار تولید می‌شود)
            metadata: اطلاعات اضافی فریم
            
        Returns:
            bool: True اگر فریم با موفقیت اضافه شد، False در غیر این صورت
        """
        if self._closed:
            warnings.warn(f"⚠️ تلاش برای push به منبع بسته شده: {self.source_id}")
            return False
        
        if not isinstance(bgr, np.ndarray):
            warnings.warn(f"⚠️ ورودی bgr باید numpy.ndarray باشد، نوع {type(bgr)} دریافت شد")
            return False
        
        if len(bgr.shape) != 3 or bgr.shape[2] != 3:
            warnings.warn(f"⚠️ شکل تصویر نامعتبر است. انتظار می‌رود (H, W, 3)، اما {bgr.shape} دریافت شد")
            return False
        
        try:
            with self._lock:
                if t_ms is None:
                    t_ms = int(time.time() * 1000)
                
                if self.auto_frame_id and frame_id is None:
                    self._frame_id += 1
                    frame_id = self._frame_id
                elif frame_id is not None:
                    self._frame_id = max(self._frame_id, frame_id)
                
                frame = VideoFrame(
                    frame_id=frame_id or self._frame_id,
                    t_ms=t_ms,
                    bgr=bgr.copy(),  # کپی برای جلوگیری از تغییرات بعدی
                    source_id=self.source_id,
                    metadata=metadata or {}
                )
                
                # تلاش برای اضافه کردن به صف
                try:
                    self._queue.put_nowait(frame)
                    success = True
                    
                except queue.Full:
                    if self.drop_oldest:
                        # حذف قدیمی‌ترین فریم
                        try:
                            _ = self._queue.get_nowait()
                            self._stats['frames_dropped'] += 1
                            # حالا دوباره سعی کن اضافه کنی
                            try:
                                self._queue.put_nowait(frame)
                                success = True
                                self._stats['queue_overflows'] += 1
                                print(f"⚠️ صف پر است! قدیمی‌ترین فریم حذف شد. (source: {self.source_id})")
                            except queue.Full:
                                success = False
                        except queue.Empty:
                            success = False
                    else:
                        # بدون حذف، فقط صبر کن
                        try:
                            self._queue.put(frame, timeout=0.1)
                            success = True
                        except queue.Full:
                            success = False
                
                if success:
                    self._stats['frames_pushed'] += 1
                    self._stats['last_push_time'] = time.time()
                    self._stats['queue_size_history'].append(self._queue.qsize())
                    
                    # فراخوانی callbackها
                    for callback in self._frame_callbacks:
                        try:
                            callback(frame)
                        except Exception as e:
                            print(f"⚠️ خطا در callback فریم: {e}")
                    
                    # اعلان به consumers
                    self._notify_consumers()
                    
                else:
                    self._stats['push_errors'] += 1
                    print(f"❌ ناتوان در اضافه کردن فریم به صف (پر است)")
                
                return success
                
        except Exception as e:
            self._stats['push_errors'] += 1
            print(f"❌ خطا در push فریم: {e}")
            return False

    def push_batch(self, frames: List[np.ndarray], 
                   t_ms_list: Optional[List[int]] = None,
                   metadata_list: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        ارسال دسته‌ای از فریم‌ها
        
        Returns:
            تعداد فریم‌های موفقیت‌آمیز اضافه شده
        """
        if t_ms_list is None:
            t_ms_list = [int(time.time() * 1000)] * len(frames)
        
        if metadata_list is None:
            metadata_list = [{}] * len(frames)
        
        success_count = 0
        for i, (frame, t_ms, metadata) in enumerate(zip(frames, t_ms_list, metadata_list)):
            if self.push(frame, t_ms, metadata=metadata):
                success_count += 1
            else:
                print(f"⚠️ فریم {i} اضافه نشد")
        
        return success_count

    def read(self, timeout: Optional[float] = 0.2, 
             block: bool = False) -> Optional[VideoFrame]:
        """
        خواندن یک فریم از صف
        
        Args:
            timeout: زمان انتظار برای دریافت فریم (ثانیه)
            block: اگر True باشد، تا دریافت فریم منتظر می‌ماند
            
        Returns:
            VideoFrame یا None اگر فریمی موجود نباشد
        """
        if self._closed and self._queue.empty():
            return None
        
        try:
            if block:
                frame = self._queue.get(timeout=None if timeout == 0 else timeout)
            else:
                frame = self._queue.get(timeout=timeout)
            
            with self._lock:
                self._stats['frames_popped'] += 1
                self._stats['last_read_time'] = time.time()
            
            return frame
            
        except queue.Empty:
            return None
        except Exception as e:
            print(f"⚠️ خطا در خواندن از صف: {e}")
            return None

    def read_batch(self, max_frames: int = 10, 
                   timeout: Optional[float] = 0.5) -> List[VideoFrame]:
        """
        خواندن چندین فریم به صورت دسته‌ای
        
        Returns:
            لیست فریم‌های خوانده شده
        """
        frames = []
        end_time = time.time() + timeout if timeout is not None else float('inf')
        
        while len(frames) < max_frames and time.time() < end_time:
            remaining_time = max(0, end_time - time.time()) if timeout is not None else timeout
            frame = self.read(timeout=remaining_time, block=False)
            if frame is None:
                break
            frames.append(frame)
        
        return frames

    def peek(self) -> Optional[VideoFrame]:
        """نگاه کردن به فریم بعدی بدون حذف آن از صف"""
        try:
            # از ویژگی داخلی صف استفاده می‌کنیم
            with self._queue.mutex:
                if self._queue._qsize() > 0:
                    return self._queue._queue[0]
            return None
        except Exception:
            return None

    def clear(self) -> None:
        """پاک کردن تمام فریم‌های موجود در صف"""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            print(f"🧹 صف پاک شد (source: {self.source_id})")

    def get_queue_size(self) -> int:
        """دریافت اندازه فعلی صف"""
        return self._queue.qsize()

    def get_stats(self) -> Dict[str, Any]:
        """دریافت آمار عملکرد"""
        with self._lock:
            stats = self._stats.copy()
            stats.update({
                'current_queue_size': self._queue.qsize(),
                'queue_capacity': self.max_queue,
                'queue_usage_percent': (self._queue.qsize() / self.max_queue) * 100,
                'is_closed': self._closed,
                'avg_queue_size': np.mean(list(self._stats['queue_size_history'])) if self._stats['queue_size_history'] else 0,
            })
            return stats

    def print_stats(self) -> None:
        """چاپ آمار عملکرد"""
        stats = self.get_stats()
        print("\n📊 آمار PushFrameSource:")
        print(f"   Source ID: {self.source_id}")
        print(f"   فریم‌های ارسال شده: {stats['frames_pushed']}")
        print(f"   فریم‌های خوانده شده: {stats['frames_popped']}")
        print(f"   فریم‌های حذف شده: {stats['frames_dropped']}")
        print(f"   خطاهای ارسال: {stats['push_errors']}")
        print(f"   سرریزهای صف: {stats['queue_overflows']}")
        print(f"   اندازه فعلی صف: {stats['current_queue_size']}/{self.max_queue} ({stats['queue_usage_percent']:.1f}%)")
        print(f"   میانگین اندازه صف: {stats['avg_queue_size']:.1f}")
        print(f"   وضعیت: {'بسته' if stats['is_closed'] else 'باز'}")

    def register_callback(self, callback) -> None:
        """ثبت تابع callback برای فریم‌های جدید"""
        if callable(callback):
            self._frame_callbacks.append(callback)
            print(f"✅ Callback ثبت شد (source: {self.source_id})")

    def unregister_callback(self, callback) -> None:
        """حذف تابع callback"""
        if callback in self._frame_callbacks:
            self._frame_callbacks.remove(callback)

    def start_monitoring(self, interval: float = 5.0) -> None:
        """شروع مانیتورینگ وضعیت صف"""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        
        self._monitoring = True
        self._monitor_interval = interval
        
        def monitor():
            while self._monitoring and not self._closed:
                time.sleep(self._monitor_interval)
                if self._queue.qsize() > self.max_queue * 0.8:
                    print(f"⚠️ هشدار: صف {self.source_id} در {self._queue.qsize()}/{self.max_queue} ({self._queue.qsize()/self.max_queue*100:.1f}%)")
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
        print(f"📈 مانیتورینگ شروع شد (source: {self.source_id})")

    def stop_monitoring(self) -> None:
        """توقف مانیتورینگ"""
        self._monitoring = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None

    def _notify_consumers(self) -> None:
        """اعلان به مصرف‌کنندگان (برای گسترش در کلاس‌های فرزند)"""
        pass

    def close(self) -> None:
        """بستن منبع و پاکسازی منابع"""
        if self._closed:
            return
        
        self._closed = True
        self._monitoring = False
        
        # توقف مانیتورینگ
        self.stop_monitoring()
        
        # پاک کردن صف
        self.clear()
        
        # پاک کردن callbackها
        self._frame_callbacks.clear()
        
        print(f"✅ PushFrameSource بسته شد (source: {self.source_id})")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __len__(self) -> int:
        """تعداد فریم‌های موجود در صف"""
        return self._queue.qsize()
    
    def __bool__(self) -> bool:
        """آیا صف خالی است؟"""
        return not self._queue.empty()


@dataclass
class BufferedPushFrameSource(PushFrameSource):
    """نسخه‌ای از PushFrameSource با بافر اضافی برای کاربردهای real-time"""
    
    buffer_size: int = 50  # اندازه بافر اضافی
    smoothing: bool = True  # هموارسازی نرخ فریم
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self._buffer = deque(maxlen=self.buffer_size)
        self._last_frame_time = 0
        self._frame_times = deque(maxlen=30)
        
    def push(self, bgr: np.ndarray, **kwargs) -> bool:
        result = super().push(bgr, **kwargs)
        if result and self.smoothing:
            current_time = time.time()
            if self._last_frame_time > 0:
                self._frame_times.append(current_time - self._last_frame_time)
            self._last_frame_time = current_time
        return result
    
    def read(self, **kwargs) -> Optional[VideoFrame]:
        # اگر بافر داشته باشیم، ابتدا از بافر بخوان
        if self._buffer:
            frame = self._buffer.popleft()
            # همچنین از صف اصلی هم یک فریم به بافر اضافه کن
            try:
                new_frame = self._queue.get_nowait()
                self._buffer.append(new_frame)
            except queue.Empty:
                pass
            return frame
        
        # در غیر این صورت از صف اصلی بخوان
        return super().read(**kwargs)
    
    def fill_buffer(self, target_size: Optional[int] = None) -> int:
        """پر کردن بافر تا اندازه مشخص"""
        if target_size is None:
            target_size = self.buffer_size
        
        filled = 0
        while len(self._buffer) < target_size:
            frame = super().read(timeout=0.1)
            if frame is None:
                break
            self._buffer.append(frame)
            filled += 1
        
        return filled


# تابع کمکی برای ایجاد منبع push
def create_push_source(source_id: str = "push0", **kwargs) -> PushFrameSource:
    """ایجاد یک منبع push با تنظیمات پیش‌فرض"""
    return PushFrameSource(source_id=source_id, **kwargs)
