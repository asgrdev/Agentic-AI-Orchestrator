# granite_client/base_client.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """خروجی استاندارد از هر LLM backend"""
    text: str                          # متن خام پاسخ
    reasoning: Optional[str] = None    # CoT اگر وجود داشت
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BaseGraniteClient(ABC):
    """
    interface مشترک برای همه backend های LLM.
    هر backend (MLX، Ollama، vLLM) باید این را implement کند.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        context: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        ساده‌ترین متد - یک پرامپت، یک پاسخ.
        Sync (blocking).
        """
        ...

    @abstractmethod
    def reason_step(
        self,
        question: str,
        context: str,
        *,
        step_hint: str = "",
    ) -> LLMResponse:
        """
        برای Chain-of-Thought reasoning.
        context = اطلاعات graph + vector
        step_hint = راهنمای اختیاری برای این مرحله
        """
        ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """بک‌اندها می‌توانند override کنند."""
        return False

    async def close(self):
        """اختیاری — بک‌اندهایی که فایل/pipe باز می‌کنند این را override می‌کنند."""
        pass

    def extract_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        context: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """
        generate() را صدا می‌زند و JSON را parse می‌کند.
        در صورت شکست، retry می‌کند.
        """
        import json
        import re

        _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

        response = self.generate(
            prompt,
            system=system,
            context=context,
            max_tokens=max_tokens,
        )
        text = response.text

        # تلاش اول
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        # تلاش دوم - repair prompt
        repair_prompt = (
            "The following text should be valid JSON. "
            "Fix any errors and return ONLY the JSON object:\n\n"
            f"{text}"
        )
        response2 = self.generate(
            repair_prompt,
            system="Return ONLY valid JSON. No explanation.",
            max_tokens=max_tokens or 512,
        )
        m2 = _JSON_RE.search(response2.text)
        if m2:
            try:
                return json.loads(m2.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"JSON extraction failed after retry.\nLast output: {response2.text[:300]}"
        )
