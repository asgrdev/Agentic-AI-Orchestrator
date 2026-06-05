from llama_cpp import Llama
from .base_client import BaseGraniteClient, LLMResponse

class LlamaCppGraniteClient(BaseGraniteClient):
    def __init__(self, model_path: str, n_ctx: int = 4096, **kwargs):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.kwargs = kwargs
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            **kwargs,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def close(self):
        # برای llama.cpp لازم است
        if hasattr(self.llm, "close"):
            self.llm.close()

    def generate(self, prompt: str, *, system=None, context=None, max_tokens=None):
        full_prompt = (system or "") + "\n" + (context or "") + "\n" + prompt
        out = self.llm(
            full_prompt,
            max_tokens=max_tokens or 512,
            echo=False,
        )
        text = out["choices"][0]["text"]
        return LLMResponse(text=text)

    def reason_step(self, question, context, *, step_hint=""):
        prompt = (
            f"Question: {question}\n"
            f"{context}\n"
            f"Hint: {step_hint}\n"
            "Think step by step.\n"
            "REASONING:\n"
        )
        out = self.llm(prompt, max_tokens=1024)
        text = out["choices"][0]["text"]

        # مشابه MLX
        # می‌توانیم همان split_reasoning را reuse کنیم
        reasoning, answer = MlxGraniteClient._split_reasoning(text)
        return LLMResponse(text=answer, reasoning=reasoning)
