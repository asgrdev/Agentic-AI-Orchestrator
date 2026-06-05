# granite_client/mlx_client.py
from __future__ import annotations

from typing import Optional

from .base_client import BaseGraniteClient, LLMResponse
from .config import MlxGraniteConfig
from .prompts import (
    SYSTEM_DEFAULT,
    SYSTEM_REASONING,
    build_prompt,
)


class MlxGraniteClient(BaseGraniteClient):
    """
    MLX backend برای Apple Silicon.
    مدل یک بار لود می‌شود و در حافظه می‌ماند.
    """

    def __init__(self, cfg: MlxGraniteConfig = MlxGraniteConfig()) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._generate_fn = None
        self._load_model()
   
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await  self.close()
        
    async def close(self):
        # MLX نیازی به free manual ندارد
        return False    
    # ─── Private: model loading ──────────────────────────────────
 

    def _load_model(self) -> None:
        try:
            from mlx_lm import load, generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:
            raise RuntimeError(
                "MLX-LM not installed. Run: pip install mlx-lm"
            ) from e
    
        self._model, self._tokenizer = load(self.cfg.model_path)
        self._generate_fn = generate
    
        # sampler یک‌بار ساخته می‌شود
        self._sampler = make_sampler(
            temp=self.cfg.temperature,
            top_p=self.cfg.top_p,
            min_p=0.0,
            min_tokens_to_keep=1,
            top_k=0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
            xtc_special_tokens=[],
        )
    
    
    def _build_kwargs(self, max_tokens: int | None) -> dict:
            """
            kwargs ای که مستقیم به mlx_lm.generate می‌رود.
            فقط آرگومان‌های تضمینی (بدون sampler object) استفاده می‌شود.
            """
            return {
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "sampler": self._sampler,
            }  
         
        

    def _call_model(self, full_prompt: str, max_tokens: int | None) -> str:
        """
        مستقیم mlx_lm.generate را صدا می‌زند.
        خروجی: متن خام (str).
        """
        import mlx.core as mx

        if self.cfg.seed is not None:
            mx.random.seed(self.cfg.seed)

        kwargs = self._build_kwargs(max_tokens)

        try:
            out = self._generate_fn(
                self._model,
                self._tokenizer,
                prompt=full_prompt,
                **kwargs,
            )
        except TypeError:
            # fallback: بدون kwargs اضافی
            out = self._generate_fn(
                self._model,
                self._tokenizer,
                prompt=full_prompt,
            )

        # mlx_lm.generate ممکن است str یا dict برگرداند
        if isinstance(out, dict):
            return str(out.get("text", ""))
        return str(out)

    # ─── Public interface ─────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        context: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        ساده‌ترین متد - یک پرامپت → یک پاسخ.

        مثال استفاده:
            response = client.generate("چه کسی اختراع تلفن را کرد؟")
            print(response.text)
        """
        full_prompt = build_prompt(
            user_message=prompt,
            system=system or SYSTEM_DEFAULT,
            context=context,
        )
        text = self._call_model(full_prompt, max_tokens)
        return LLMResponse(text=text.strip())

    def reason_step(
        self,
        question: str,
        context: str,
        *,
        step_hint: str = "",
    ) -> LLMResponse:
        """
        برای Chain-of-Thought reasoning در LLMReasoner.

        ورودی:
            question  : سوال اصلی کاربر
            context   : اطلاعات graph + vector (از ContextBuilder)
            step_hint : مثلاً "focus on temporal relations"

        خروجی LLMResponse:
            .text      : پاسخ نهایی
            .reasoning : مراحل فکری (اگر مدل تولید کند)

        مثال:
            resp = client.reason_step(
                question="کی تاسیس شد؟",
                context="[Graph]\nNode: شرکت X ...\n[Docs]\n[1] ...",
                step_hint="focus on founding dates",
            )
        """
        instruction = (
            f"Question: {question}\n\n"
            f"{'Hint: ' + step_hint + chr(10) if step_hint else ''}"
            "Instructions:\n"
            "1. Read the context carefully.\n"
            "2. Think step-by-step (write your reasoning).\n"
            "3. Then write your final answer after 'ANSWER:'.\n\n"
            "Format:\n"
            "REASONING: <your step-by-step thoughts>\n"
            "ANSWER: <your final answer with [N] citations>"
        )

        full_prompt = build_prompt(
            user_message=instruction,
            system=SYSTEM_REASONING,
            context=context,
        )

        raw_text = self._call_model(full_prompt, max_tokens=None)

        # جداسازی reasoning از answer
        reasoning, answer = self._split_reasoning(raw_text)

        return LLMResponse(
            text=answer.strip(),
            reasoning=reasoning.strip() if reasoning else None,
        )

    # ─── Helper ───────────────────────────────────────────────────

    @staticmethod
    def _split_reasoning(raw: str) -> tuple[str, str]:
        """
        متن را به دو بخش REASONING و ANSWER تقسیم می‌کند.

        مثال ورودی:
            "REASONING: first I looked at...
             ANSWER: The company was founded in 1990."

        خروجی: ("first I looked at...", "The company was founded in 1990.")
        """
        import re

        reasoning = ""
        answer = raw

        r_match = re.search(r"REASONING\s*:\s*(.*?)(?=ANSWER\s*:|$)", raw, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r"ANSWER\s*:\s*(.*)", raw, re.DOTALL | re.IGNORECASE)

        if r_match:
            reasoning = r_match.group(1).strip()
        if a_match:
            answer = a_match.group(1).strip()

        return reasoning, answer
