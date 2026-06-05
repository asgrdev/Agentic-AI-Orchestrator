from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
import warnings
import time

try:
    from runtime_module_io_logger import log_module_io
except Exception:  # pragma: no cover
    def log_module_io(*args: Any, **kwargs: Any) -> None:
        return


# ---- top-level split wiring: efficientat ----
from .efficientat_parts.panns_processor import PANNsProcessor
# ============================================================================
# معماری CNN14 برای PANNs - نسخه ساده شده
# ============================================================================

class ConvBlock(nn.Module):
    """بلوک کانولوشن برای PANNs"""
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)

    def forward(self, input: torch.Tensor, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        else:
            x = F.avg_pool2d(x, kernel_size=pool_size)
        
        return x

class Cnn14Simple(nn.Module):
    """معماری ساده شده CNN14 برای PANNs"""
    def __init__(self, sample_rate: int = 32000, classes_num: int = 527):
        super(Cnn14Simple, self).__init__()
        
        self.classes_num = classes_num
        self.sample_rate = sample_rate
        
        # Mel-spectrogram extractor
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=64,
            f_min=50,
            f_max=14000
        )
        
        # Conv blocks
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        # Adaptive pooling برای مدیریت اندازه‌های مختلف
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc_audioset.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc_audioset.bias, 0)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # accept (B,T) or (B,1,T)
        if input.dim() == 2:
            # (B,T) -> (B,1,T) for torchaudio MelSpectrogram
            input = input.unsqueeze(1)
        elif input.dim() == 3:
            # already (B,1,T) ok
            pass
        else:
            raise ValueError(f"Expected (B,T) or (B,1,T), got {tuple(input.shape)}")
    
        x = self.mel_extractor(input)     # (B, n_mels, frames) OR (B,1,n_mels,frames) depending on torchaudio version
    
        # torchaudio MelSpectrogram usually returns (B, n_mels, frames)
        if x.dim() == 4:
            # if it returned (B,1,n_mels,frames) collapse channel
            x = x.squeeze(1)
    
        x = torch.log(x + 1e-10)          # (B, n_mels, frames)
        x = x.unsqueeze(1)                # (B, 1, n_mels, frames) -> conv2d expects 4D
    
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout(x)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout(x)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout(x)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout(x)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.dropout(x)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = self.dropout(x)
    
        x = self.adaptive_pool(x)         # (B,2048,1,1)
        x = x.view(x.size(0), -1)         # (B,2048)
    
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_audioset(x)
        return x

# ============================================================================
# کانفیگ کلاس
# ============================================================================

@dataclass
class PANNsConfig:
    """
    کانفیگ PANNs Processor مشابه EfficientATConfig
    """
    # تنظیمات مدل
    model_name: str = "CNN14"  # CNN6, CNN10, CNN14
    model_path: str = ""  # مسیر local model (اختیاری)
    labels_path: str = ""  # مسیر فایل labels (یک label در هر خط)
    device: str = "cpu"  # cpu یا cuda
    topk: int = 8  # تعداد top-k classes
    
    # تنظیمات پردازش صوتی
    sample_rate: int = 32000  # PANNs از 32kHz استفاده می‌کند
    target_length: int = 32000  # 1 ثانیه در 32kHz
    num_classes: int = 527  # تعداد کلاس‌های AudioSet
    
    # تنظیمات پیش‌پردازش
    normalize_audio: bool = True  # نرمالیزاسیون amplitude
    resample_input: bool = True  # تبدیل sample rate اگر لازم باشد
    use_waveform: bool = True  # PANNs مستقیماً waveform می‌گیرد
    
    # تنظیمات performance
    batch_size: int = 1  # برای CPU بهتر است 1 باشد
    enable_cache: bool = True  # کش کردن نتایج
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # فیلدهای داخلی (غیرقابل تنظیم توسط کاربر)
    _labels: List[str] = field(default_factory=list, init=False, repr=False)
    _label_to_idx: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _idx_to_label: Dict[int, str] = field(default_factory=dict, init=False, repr=False)
    
    def __post_init__(self):
        """اعتبارسنجی و مقداردهی اولیه"""
        # اعتبارسنجی device
        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # اعتبارسنجی model_name
        valid_models = ["CNN6", "CNN10", "CNN14"]
        if self.model_name not in valid_models:
            raise ValueError(f"model_name must be one of {valid_models}, got {self.model_name}")
        
        # بارگذاری labels اگر فایل وجود دارد
        if self.labels_path and Path(self.labels_path).exists():
            self._load_labels()
    
    def _load_labels(self):
        """بارگذاری labels از فایل"""
        try:
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self._labels = [line.strip() for line in f if line.strip()]
            
            # ایجاد mapping
            self._label_to_idx = {label: idx for idx, label in enumerate(self._labels)}
            self._idx_to_label = {idx: label for idx, label in enumerate(self._labels)}
            
            print(f"[Config] Loaded {len(self._labels)} labels from {self.labels_path}")
            
            # اعتبارسنجی تعداد labels
            if len(self._labels) != self.num_classes:
                warnings.warn(f"Number of labels ({len(self._labels)}) doesn't match "
                            f"num_classes ({self.num_classes})")
                
        except Exception as e:
            warnings.warn(f"Failed to load labels from {self.labels_path}: {e}")
    
    def get_label(self, idx: int) -> str:
        """دریافت label بر اساس اندیس"""
        if self._idx_to_label:
            return self._idx_to_label.get(idx, f"Class_{idx}")
        return f"Class_{idx}"
    
    def get_idx(self, label: str) -> int:
        """دریافت اندیس بر اساس label"""
        return self._label_to_idx.get(label, -1)
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به dictionary"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "num_classes": self.num_classes,
            "topk": self.topk,
            "labels_loaded": len(self._labels) > 0,
            "total_labels": len(self._labels)
        }

# ============================================================================
# کلاس اصلی PANNsProcessor
# ============================================================================


# ============================================================================
# Factory function برای backward compatibility
# ============================================================================

def create_panns_processor(
    model_name: str = "CNN14",
    device: str = "cpu",
    labels_path: str = "",
    **kwargs
) -> PANNsProcessor:
    """
    تابع کمکی برای ایجاد PANNsProcessor
    
    Args:
        model_name: نام مدل
        device: دستگاه پردازش
        labels_path: مسیر فایل labels
        **kwargs: پارامترهای اضافی برای PANNsConfig
        
    Returns:
        نمونه PANNsProcessor
    """
    config = PANNsConfig(
        model_name=model_name,
        device=device,
        labels_path=labels_path,
        **kwargs
    )
    
    return PANNsProcessor(config)
