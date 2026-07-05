# ... imports ...
from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import logging
import re

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

@dataclass(frozen=True)
class MlxGraniteConfig:
    model_path: str = "/Users/dbk/Desktop/RAG/models/granite4_3b"#"granite4-7b-instruct"
    max_tokens: int = 1500
    temperature: float = 0.74
    top_p: float = 0.94
    seed: int = 0
    stop: Optional[list[str]] = None

class MlxGraniteAnswerGenerator:
   

    def __init__(self, cfg: MlxGraniteConfig) -> None:
        self.cfg = cfg
        try:
            from mlx_lm import load, stream_generate, generate
            from mlx_lm.sample_utils import make_sampler
            from transformers import PreTrainedTokenizer
        except Exception as e:
            raise RuntimeError("Install mlx-lm and transformers: pip install mlx-lm transformers") from e
        
        self._load = load
        self._generate = generate
        self._stream_generate = stream_generate

        # لود lazy از طریق ModelGate — قبلاً این کلاس granite را مستقیم لود
        # می‌کرد و در کنار phi3 و embedding روی GPU می‌ماند → METAL OOM
        from core.model_gate import get_model_gate
        self._gate = get_model_gate()
        self._model_key = f"mlx:{cfg.model_path}"
        _path = cfg.model_path
        self._model_loader = lambda: self._load(_path)

        # make_sampler signature varies across mlx-lm versions; fall back to a
        # simple argmax sampler only if it rejects these arguments
        try:
            self._sampler = make_sampler(
                temp=cfg.temperature,
                top_p=cfg.top_p,
                min_p=0.0,
                min_tokens_to_keep=1,
                top_k=0,
                xtc_probability=0.0,
                xtc_threshold=0.0,
                xtc_special_tokens=[],
            )
        except TypeError:
            self._sampler = lambda logits: self._simple_sampler(
                logits, cfg.temperature, cfg.top_p
            )

    def _simple_sampler(self, logits, temperature, top_p):
        # Simple top-p sampling implementation if make_sampler is unavailable
        import mlx.core as mx
        logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        # Top-P logic simplified: just argmax for now if you don't have the full utils
        # Or better, use the standard argmax if you just want to avoid the error:
        return mx.argmax(logits, axis=-1)

    def _build_gen_kwargs(self) -> dict:
        """Build kwargs strictly compatible with mlx_lm.generate_step signature."""
        # The generate_step function signature from your code:
        # def generate_step(prompt, model, *, max_tokens, sampler, ..., seed=None, ...)
        
        # CRITICAL: 'seed' is NOT a direct argument in generate_step in the provided code.
        # You must set it in the global mlx.random.seed() before calling generate.
        
        kwargs = {
            "max_tokens": self.cfg.max_tokens,
            "sampler": self._sampler,
            "kv_bits": 8,       # فشرده‌سازی KV Cache به ۱۶ بیت (کاهش شدید مصرف رم)
            "kv_group_size": 32, 
            "quantized_kv_start": 0 # شروع فشرده‌سازی از همان ابتدا
            # Do NOT pass 'temperature', 'top_p', or 'seed' directly here.
        }
        if self.cfg.stop:
            # Note: 'stop' is not a direct arg in generate_step either, 
            # but usually handled via eos_token_ids in the tokenizer or sampler.
            # If your wrapper expects 'stop', you might need to override the tokenizer's eos_tokens.
            pass 
            
        return kwargs

    def extract_answer(self, *, prompt: str, ) -> Dict[str, Any]:
        import mlx.core as mx
        
        sys = (
             "You are Granite, a reliable multimodal assistant for a live perception system. "
             "Answer in the same language as the user unless asked otherwise. "
             "Prioritize a direct useful answer first. "
             "Use only grounded context and do not invent facts. "
             "Do not output punctuation-only text, malformed fragments, or one-word answers."
        )
        full = f"SYSTEM: {sys}\n\nUSER: {prompt}\n\nASSISTANT:"
        
        # Set seed if needed (if your version of mlx-lm supports it via global seed)
        if self.cfg.seed is not None:
            mx.random.seed(self.cfg.seed)

        # acquire داخل slot — در حالت serial بین لود و اجرا، مدل دیگری لود نمی‌شود
        with self._gate.execution_slot(self._model_key):
            model, tokenizer = self._gate.acquire(
                key=self._model_key, kind="llm", loader=self._model_loader
            )
            try:
                out = self._generate(model, tokenizer, prompt=full,
                                     max_kv_size=2048,  # محدود کردن سایز حافظه KV
                                     **self._build_gen_kwargs())
            except TypeError as e:
                # Fallback: try without extra kwargs if sampler is the issue
                logger.warning("mlx_lm.generate rejected kwargs (%s) — retrying plain", e)
                out = self._generate(model, tokenizer, full)
        return out
        #return self._process_output(out, full, sys, schema_hint)

    def _process_output(self, out, prompt, sys, schema_hint):
        text = str(out.get("text", out) if isinstance(out, dict) else out)
        js = self._parse_json(text)
        if js is not None:
            return js

        # Retry logic
        repair_prompt = "Fix the following into STRICT valid JSON matching the schema. Return ONLY JSON.\n\nTEXT:\n" + text
        repair_full = f"SYSTEM: {sys}\n\nUSER: {repair_prompt}\n\nASSISTANT:"

        with self._gate.execution_slot(self._model_key):
            model, tokenizer = self._gate.acquire(
                key=self._model_key, kind="llm", loader=self._model_loader
            )
            try:
                out2 = self._generate(model, tokenizer, repair_full, **self._build_gen_kwargs())
            except TypeError:
                out2 = self._generate(model, tokenizer, repair_full)
            
        text2 = str(out2.get("text", out2) if isinstance(out2, dict) else out2)
        js2 = self._parse_json(text2)
        if js2 is not None:
            return js2

        raise ValueError("Granite JSON extraction failed")

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        m = _JSON_RE.search(text)
        if not m:
            return None
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            return None
