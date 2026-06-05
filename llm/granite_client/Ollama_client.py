import ollama
from .base_client import BaseGraniteClient, LLMResponse
from .prompts import build_prompt, SYSTEM_DEFAULT, SYSTEM_REASONING

class OllamaGraniteClient(BaseGraniteClient):
    def __init__(self, model_name: str = "granite", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False  # no resources

    async def close(self):
        return

    def generate(self, prompt, *, system=None, context=None, max_tokens=None):
        full_prompt = build_prompt(
            user_message=prompt, system=system or SYSTEM_DEFAULT, context=context
        )

        resp = ollama.generate(
            model=self.model_name,
            prompt=full_prompt,
            options={"num_predict": max_tokens} if max_tokens else None,
        )
        return LLMResponse(text=resp["response"].strip())

    def reason_step(self, question, context, *, step_hint=""):
        prompt = (
            f"Question: {question}\n"
            f"{context}\n"
            f"Hint: {step_hint}\n"
            "Think step-by-step.\n"
            "REASONING:\n"
        )

        resp = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            options={"num_predict": 1024},
        )
        raw = resp["response"]

        reasoning, answer = MlxGraniteClient._split_reasoning(raw)
        return LLMResponse(text=answer, reasoning=reasoning)
