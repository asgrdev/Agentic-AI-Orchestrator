from __future__ import annotations
from .. import efficientat as _src_mod
globals().update({k: v for k, v in vars(_src_mod).items() if not k.startswith("__")})

class PANNsProcessor:
    """
    پردازنده PANNs با اینترفیس مشابه EfficientATProcessor
    """
    
    def __init__(self, cfg: PANNsConfig) -> None:
        """
        مقداردهی اولیه
        
        Args:
            cfg: کانفیگ PANNs
        """
        self.cfg = cfg
        
        # دستگاه پردازش
        self.device = torch.device(cfg.device)
        
        # لاگ اطلاعات
        self._log("INFO", f"Initializing PANNsProcessor with {cfg.model_name}")
        self._log("INFO", f"Device: {self.device}")
        
        # بارگذاری مدل
        self.model = self._load_model()
        
        # کش برای نتایج
        self._cache = {}
        self.last_logits = None
        self.last_embedding = None
        self.last_probs = None
        
        # آمار عملکرد
        self.stats = {
            "total_inferences": 0,
            "total_samples": 0,
            "avg_inference_time": 0.0,
        }
        
        self._log("INFO", "PANNsProcessor initialized successfully")
        log_module_io(
            module="panns.audio_tagger",
            stage="model_load",
            inputs={
                "model_name": cfg.model_name,
                "device": cfg.device,
                "sample_rate": int(cfg.sample_rate),
                "target_length": int(cfg.target_length),
                "topk": int(cfg.topk),
                "labels_path": cfg.labels_path,
            },
            outputs={
                "loaded": True,
                "num_classes": int(cfg.num_classes),
                "labels_loaded": int(len(cfg._labels)),
            },
        )
    
    def _log(self, level: str, message: str):
        """لاگ پیام با سطح مشخص"""
        log_levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        cfg_level = log_levels.get(self.cfg.log_level.upper(), 1)
        msg_level = log_levels.get(level.upper(), 1)

        if msg_level >= cfg_level:
            print(f"[PANNs {level}] {message}")

    def _resolve_cnn14_class(self):
        """
        Resolve CNN14 class safely under circular-import conditions.
        """
        cls = globals().get("Cnn14Simple")
        if cls is not None:
            return cls
        try:
            # Late import avoids NameError when this module is imported
            # before efficientat finishes defining Cnn14Simple.
            from ..efficientat import Cnn14Simple as _Cnn14Simple
            globals()["Cnn14Simple"] = _Cnn14Simple
            return _Cnn14Simple
        except Exception as e:
            raise RuntimeError(f"Cnn14Simple_unavailable: {e}") from e

    def _create_cnn14_architecture(self) -> nn.Module:
        """ایجاد معماری CNN14 ساده شده"""
        self._log("INFO", "Creating CNN14Simple architecture")
        cnn14_cls = self._resolve_cnn14_class()
        return cnn14_cls(
            sample_rate=self.cfg.sample_rate,
            classes_num=self.cfg.num_classes
        )

    @staticmethod
    def _normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize common checkpoint prefixes (e.g. DataParallel/module wrappers).
        """
        out: Dict[str, Any] = {}
        for k, v in dict(state_dict or {}).items():
            nk = str(k)
            for prefix in ("module.", "model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            out[nk] = v
        return out

    @staticmethod
    def _is_non_critical_missing_key(key: str) -> bool:
        """
        Keys that are generated at runtime (buffers/front-end) and are safe to miss.
        """
        return str(key) in {
            "mel_extractor.spectrogram.window",
            "mel_extractor.mel_scale.fb",
        }

    @staticmethod
    def _is_non_critical_unexpected_key(key: str) -> bool:
        """
        Known mismatch when loading official PANNs checkpoints into simplified model.
        """
        k = str(key)
        return (
            k.startswith("spectrogram_extractor.")
            or k.startswith("logmel_extractor.")
            or k.startswith("spec_augmenter.")
            or k.startswith("bn0.")
        )

    def _filter_incompatible_shapes(
        self,
        state_dict: Dict[str, Any],
        model: nn.Module,
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Drop checkpoint tensors whose shape is incompatible with target model.
        """
        model_state = model.state_dict()
        out: Dict[str, Any] = {}
        dropped: List[str] = []
        for k, v in dict(state_dict or {}).items():
            if k not in model_state:
                out[k] = v
                continue
            try:
                ckpt_shape = tuple(getattr(v, "shape", ()))
                model_shape = tuple(getattr(model_state[k], "shape", ()))
                if ckpt_shape and model_shape and ckpt_shape != model_shape:
                    dropped.append(f"{k}: ckpt{ckpt_shape} != model{model_shape}")
                    continue
            except Exception:
                pass
            out[k] = v
        return out, dropped
    
    def _load_model(self) -> nn.Module:
        """بارگذاری مدل PANNs"""
        self._log("INFO", f"Loading PANNs model: {self.cfg.model_name}")
        
        model = None
        model_name = self.cfg.model_name.upper()
        
        # ابتدا سعی می‌کنیم مدل محلی را بارگذاری کنیم (مسیر کانفیگ در اولویت است)
        configured_path = str(getattr(self.cfg, "model_path", "") or "").strip()
        local_model_path = configured_path if configured_path else "models/Cnn14.pth"
        if configured_path and not Path(local_model_path).exists():
            fallback_path = "models/Cnn14.pth"
            if Path(fallback_path).exists():
                self._log("WARNING", f"Configured model not found: {local_model_path}; fallback to {fallback_path}")
                local_model_path = fallback_path
        
        if Path(local_model_path).exists():
            try:
                self._log("INFO", f"Loading local model from: {local_model_path}")
                
                # ابتدا معماری را ایجاد می‌کنیم
                model = self._create_cnn14_architecture()
                
                # بارگذاری state_dict
                checkpoint = torch.load(local_model_path, map_location='cpu')
                
                # تشخیص ساختار فایل
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        # فرض می‌کنیم خود state_dict است
                        state_dict = checkpoint
                else:
                    # اگر فایل فقط state_dict باشد
                    state_dict = checkpoint

                state_dict = self._normalize_state_dict_keys(dict(state_dict or {}))
                state_dict, dropped_shape_keys = self._filter_incompatible_shapes(state_dict, model)
                if dropped_shape_keys:
                    self._log("INFO", f"Dropped incompatible checkpoint tensors: {dropped_shape_keys[:5]}")

                # بارگذاری وزن‌ها
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

                missing_keys = list(missing_keys or [])
                unexpected_keys = list(unexpected_keys or [])

                non_critical_missing = [k for k in missing_keys if self._is_non_critical_missing_key(k)]
                critical_missing = [k for k in missing_keys if k not in non_critical_missing]
                non_critical_unexpected = [k for k in unexpected_keys if self._is_non_critical_unexpected_key(k)]
                critical_unexpected = [k for k in unexpected_keys if k not in non_critical_unexpected]

                if non_critical_missing:
                    self._log("INFO", f"Ignored non-critical missing keys: {non_critical_missing[:5]}")
                if critical_missing:
                    self._log("WARNING", f"Missing keys: {critical_missing[:5]}...")

                if non_critical_unexpected:
                    self._log("INFO", f"Ignored non-critical unexpected keys: {non_critical_unexpected[:5]}")
                if critical_unexpected:
                    self._log("WARNING", f"Unexpected keys: {critical_unexpected[:5]}...")

                self._log("INFO", f"Successfully loaded local model from: {local_model_path}")
                
            except Exception as e:
                self._log("ERROR", f"Failed to load local model: {e}")
                self._log("INFO", "Creating fresh CNN14Simple architecture")
                model = self._create_cnn14_architecture()
        else:
            self._log("WARNING", f"Local model not found at {local_model_path}")
            model = self._create_cnn14_architecture()
        
        # انتقال به device
        model = model.to(self.device)
        model.eval()
        
        # محاسبه تعداد پارامترها
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self._log("INFO", f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
    
    def _preprocess_audio(self, pcm: np.ndarray) -> torch.Tensor:
        # to tensor float32
        if isinstance(pcm, np.ndarray):
            waveform = torch.from_numpy(pcm.astype("float32"))
        else:
            waveform = pcm.detach().float()
    
        waveform = waveform.to(self.device)
    
        # mono: (T,) or (C,T) -> (T,)
        if waveform.dim() == 2:
            # assume (C,T)
            waveform = waveform.mean(dim=0)
        elif waveform.dim() > 2:
            raise ValueError(f"Unexpected pcm shape: {tuple(waveform.shape)}")
    
        # normalize
        if self.cfg.normalize_audio:
            max_val = waveform.abs().max()
            if float(max_val) > 0:
                waveform = waveform / max_val
    
        # optional resample: IMPORTANT - you must know original_sr
        if self.cfg.resample_input:
            original_sr = getattr(self.cfg, "input_sample_rate", 16000)  # add this to config ideally
            target_sr = int(self.cfg.sample_rate)
            if int(original_sr) != target_sr:
                import torchaudio.functional as AF
                waveform = AF.resample(waveform, int(original_sr), target_sr)
    
        # pad/crop to target_length
        target_length = int(self.cfg.target_length)
        if target_length > 0:
            cur = int(waveform.numel())
            if cur > target_length:
                start = (cur - target_length) // 2
                waveform = waveform[start:start + target_length]
            elif cur < target_length:
                pad = target_length - cur
                waveform = torch.nn.functional.pad(waveform, (0, pad), value=0.0)
    
        # make batch dimension: (1, T)
        waveform = waveform.unsqueeze(0)
    
        self._log("DEBUG", f"Preprocessed waveform shape: {tuple(waveform.shape)}")  # (B,T)
        return waveform

    
    def forward_model(self, pcm: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        اجرای مدل روی PCM ورودی
        
        Args:
            pcm: آرایه numpy صوتی
            
        Returns:
            logits_np: logits خروجی
            emb_np: embedding یا None
        """
        start_time = time.time()
        
        try:
            # پیش‌پردازش
            waveform = self._preprocess_audio(pcm)
            
            # بررسی shape نهایی
            self._log("DEBUG", f"Model input shape: {waveform.shape}")
            
            # استنتاج
            with torch.no_grad():
                logits = self.model(waveform)  # خروجی اصلی
            
            self._log("DEBUG", f"Model output shape: {logits.shape}")
            
            # تبدیل به numpy
            logits_np = logits.squeeze(0).detach().float().cpu().numpy()
            
            # ذخیره در کش
            self.last_logits = logits_np
            
            # محاسبه احتمالات با sigmoid (چون multiclass multilabel است)
            self.last_probs = torch.sigmoid(torch.from_numpy(logits_np)).numpy()
            
            # به‌روزرسانی آمار
            inference_time = time.time() - start_time
            self.stats["total_inferences"] += 1
            self.stats["total_samples"] += int(waveform.size(1))
            self.stats["avg_inference_time"] = (
                (self.stats["avg_inference_time"] * (self.stats["total_inferences"] - 1) + inference_time) /
                self.stats["total_inferences"]
            )
            
            self._log("DEBUG", f"Inference time: {inference_time:.3f}s")
            
            return logits_np, None  # embedding ندارد
            
        except Exception as e:
            self._log("ERROR", f"forward_model failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def infer(self, pcm: np.ndarray, now_ms: int = 0) -> tuple:
        """
        اینترفیس اصلی برای استنتاج - مشابه EfficientATProcessor
        
        Args:
            pcm: آرایه صوتی
            now_ms: timestamp (برای compatibility)
            
        Returns:
            topk_items: لیست (label, probability)
            entropy: entropy توزیع
            conf: confidence
            embedding: embedding صوتی
        """
        t0 = time.time()
        log_module_io(
            module="panns.audio_tagger",
            stage="input",
            inputs={
                "now_ms": int(now_ms),
                "sample_count": int(pcm.shape[0]) if hasattr(pcm, "shape") else None,
                "dtype": str(getattr(pcm, "dtype", "")),
                "enable_cache": bool(self.cfg.enable_cache),
                "topk": int(self.cfg.topk),
            },
        )
        try:
            # بررسی کش
            cache_key = hash(pcm.tobytes())
            if self.cfg.enable_cache and cache_key in self._cache:
                self._log("DEBUG", "Using cached result")
                cached = self._cache[cache_key]
                topk_items, entropy, conf, embedding = cached
                log_module_io(
                    module="panns.audio_tagger",
                    stage="output",
                    outputs={
                        "cached": True,
                        "topk_items": [{"label": str(l), "prob": float(p)} for l, p in list(topk_items)[:8]],
                        "entropy": float(entropy),
                        "confidence": float(conf),
                        "embedding_present": embedding is not None,
                        "latency_ms": round((time.time() - t0) * 1000.0, 3),
                    },
                )
                return cached
            
            # استنتاج
            logits, embedding = self.forward_model(pcm)
            
            # محاسبه احتمالات
            probs = self.last_probs
            
            # محاسبه top-k
            topk = min(self.cfg.topk, len(probs))
            topk_indices = np.argsort(probs)[::-1][:topk]
            
            # ساخت لیست (label, probability)
            topk_items = []
            for idx in topk_indices:
                label = self.cfg.get_label(int(idx))
                prob = float(probs[idx])
                topk_items.append((label, prob))
            
            # محاسبه entropy
            epsilon = 1e-10
            entropy = -np.sum(probs * np.log(probs + epsilon))
            
            # محاسبه confidence (max probability)
            conf = np.max(probs)
            
            topk_items = []
            for idx in topk_indices:
                label = self.cfg.get_label(int(idx))
                prob = float(probs[idx])
                topk_items.append((label, prob))

            out = (topk_items, entropy, conf, embedding)
            if self.cfg.enable_cache:
                self._cache[cache_key] = out
            log_module_io(
                module="panns.audio_tagger",
                stage="output",
                outputs={
                    "cached": False,
                    "topk_items": [{"label": str(l), "prob": float(p)} for l, p in topk_items[:8]],
                    "entropy": float(entropy),
                    "confidence": float(conf),
                    "embedding_present": embedding is not None,
                    "latency_ms": round((time.time() - t0) * 1000.0, 3),
                },
            )
            # تغییر signature برگشتی
            return out
        
        except Exception as e:
            self._log("ERROR", f"infer failed: {e}")
            log_module_io(
                module="panns.audio_tagger",
                stage="error",
                error=repr(e),
                metadata={"latency_ms": round((time.time() - t0) * 1000.0, 3)},
            )
            return [], 0.0, 0.0, None
    
    def get_topk_with_probabilities(self, pcm: np.ndarray) -> List[Tuple[str, float]]:
        """
        دریافت top-k labelها با احتمالات
        
        Args:
            pcm: آرایه صوتی
            
        Returns:
            لیستی از tupleهای (label, probability)
        """
        try:
            # استنتاج
            topk_items, _, _, _ = self.infer(pcm)
            return topk_items[:self.cfg.topk]
            
        except Exception as e:
            self._log("ERROR", f"get_topk_with_probabilities failed: {e}")
            return []
    
    def get_labels_for_indices(self, indices: np.ndarray) -> List[str]:
        """
        دریافت labelهای متنی برای اندیس‌ها
        
        Args:
            indices: آرایه اندیس‌ها
            
        Returns:
            لیست labelها
        """
        labels = []
        for idx in indices:
            if idx < len(self.cfg._labels):
                labels.append(self.cfg._labels[idx])
            else:
                labels.append(f"Class_{idx}")
        return labels
    
    def get_model_info(self) -> Dict[str, Any]:
        """دریافت اطلاعات مدل"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "model_name": self.cfg.model_name,
            "model_type": "PANNs",
            "device": str(self.device),
            "sample_rate": self.cfg.sample_rate,
            "num_classes": self.cfg.num_classes,
            "total_params": f"{total_params:,}",
            "input_shape": "waveform: [batch, 1, time]",
            "labels_loaded": len(self.cfg._labels) > 0,
            "total_labels": len(self.cfg._labels),
            "stats": self.stats.copy()
        }
    
    def clear_cache(self):
        """پاک کردن کش"""
        self._cache.clear()
        self._log("INFO", "Cache cleared")
    
    def reset_stats(self):
        """بازنشانی آمار"""
        self.stats = {
            "total_inferences": 0,
            "total_samples": 0,
            "avg_inference_time": 0.0,
        }
        self._log("INFO", "Statistics reset")
