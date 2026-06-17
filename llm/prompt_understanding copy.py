# llm/phi4_mini_client.py
from __future__ import annotations

import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from llm.tool_registry import TOOL_REGISTRY

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

UNDERSTAND_SYSTEM = """\
You are a precise query understanding engine embedded in an AgenticGraphRAG system.

Your responsibilities:
1. Detect intent: factual | analytical | comparative | exploratory | procedural
2. Extract named entities with types: PERSON | ORG | LOC | CONCEPT | EVENT | DATE | PRODUCT
3. Decompose complex queries into ordered atomic sub-questions
4. Assign a complexity score (1=simple, 5=very complex)
5. Select ONLY relevant tools from the registry — do NOT hallucinate tool names

Rules:
- Respond ONLY with valid JSON, no markdown, no explanation
- If no tools are needed, return tool_calls as []
- sub_questions must be self-contained and answerable independently
"""

UNDERSTAND_TEMPLATE = """\
Query: {query}
Conversation history (last 3 turns): {history}
Available tools: {tools}

Respond with this exact JSON schema:
{{
  "intent": "factual|analytical|comparative|exploratory|procedural",
  "complexity": 1,
  "language": "en|fa|...",
  "entities": [
    {{"text": "...", "type": "PERSON|ORG|LOC|CONCEPT|EVENT|DATE|PRODUCT", "relevance": 0.9}}
  ],
  "sub_questions": ["..."],
  "tool_calls": [
    {{"tool": "...", "args": {{}}, "reason": "why this tool"}}
  ],
  "search_keywords": ["..."],
  "requires_realtime": false
}}
"""

PLAN_SYSTEM = """\
You are a task planner for an agentic RAG system.
Given a query and its understanding, produce an ordered execution plan.
Each step must reference a concrete action: retrieve | reason | search | summarize | compare | validate.
Respond ONLY with valid JSON.
"""

PLAN_TEMPLATE = """\
Query: {query}
Understanding: {understanding}

Produce a plan:
{{
  "steps": [
    {{
      "id": 1,
      "action": "retrieve|reason|search|summarize|compare|validate",
      "description": "...",
      "depends_on": [],
      "tool": "tool_name_or_null",
      "args": {{}}
    }}
  ],
  "estimated_hops": 2,
  "strategy": "sequential|parallel|iterative"
}}
"""

GAP_SYSTEM = """\
You are a knowledge gap detector for a RAG pipeline.
Analyze whether the provided context is sufficient to answer the query.
Respond ONLY with valid JSON.
"""

GAP_TEMPLATE = """\
Query: {query}
Retrieved context:
{context}

{{
  "sufficient": true,
  "confidence": 0.85,
  "missing_aspects": ["..."],
  "search_queries": ["..."],
  "suggested_sources": ["web|graph|vector|kg"]
}}
"""



# ─────────────────────────────────────────────
# Backend Abstraction
# ─────────────────────────────────────────────

class LLMBackend(ABC):
    """Abstract base — هر backend باید این را پیاده کند"""

    @abstractmethod
    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


# ── 1. Ollama ──────────────────────────────────

class OllamaBackend(LLMBackend):
    """
    Ollama REST API  →  /api/chat
    مناسب برای اجرای محلی روی هر پلتفرم
    """

    def __init__(self, base_url: str, model: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def close(self) -> None:
        await self._client.aclose()


# ── 2. llama.cpp (llama-cpp-python) ───────────

class LlamaCppBackend(LLMBackend):
    """
    llama-cpp-python  →  اجرای مستقیم GGUF روی CPU/GPU
    نیاز به: pip install llama-cpp-python
    
    دو حالت:
      a) server=True  → HTTP server روی پورت محلی (OpenAI-compatible)
      b) server=False → بارگذاری مستقیم در پروسه (in-process)
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        server_url: str | None = None,
    ):
        self._server_url = server_url  # اگر llama.cpp server جداگانه اجرا شده
        self._llm = None

        if not server_url:
            # in-process load
            try:
                from llama_cpp import Llama  # type: ignore
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False,
                )
            except ImportError as e:
                raise ImportError(
                    "llama-cpp-python not installed: pip install llama-cpp-python"
                ) from e
        else:
            self._http = httpx.AsyncClient(timeout=60)

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        if self._server_url:
            # OpenAI-compatible endpoint که llama.cpp server expose می‌کند
            payload = {
                "model": "local",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            resp = await self._http.post(
                f"{self._server_url}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        # in-process — synchronous call را در thread pool اجرا می‌کنیم
        import asyncio
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._llm(prompt, max_tokens=max_tokens, temperature=temperature),
        )
        return result["choices"][0]["text"]

    async def close(self) -> None:
        if hasattr(self, "_http"):
            await self._http.aclose()


# ── 3. MLX / mlx_lm (Apple Silicon) ──────────

class MLXBackend(LLMBackend):
    """
    mlx_lm  →  اجرای بهینه روی Apple Silicon (M1/M2/M3/M4)
    نیاز به: pip install mlx-lm transformers

    پارامترهای sampling از طریق make_sampler به generate پاس می‌شوند.
    generation همزمان (sync) است → در executor اجرا می‌شود.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        kv_bits: int = 8,
        kv_group_size: int = 32,
    ):
        try:
            from mlx_lm import load, generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:
            raise ImportError(
                "mlx-lm not installed: pip install mlx-lm transformers"
            ) from e
        import os

        # Set the environment variable before importing MLX
        os.environ["MLX_GPU"] = "0"
        self._generate = generate
        self.model, self.tokenizer = load(model_path)
        logger.info(f"MLX model loaded: {model_path}")

        # sampler یک‌بار ساخته می‌شود — overhead تکراری ندارد
        self._sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=0.0,
            min_tokens_to_keep=1,
            top_k=0,
            xtc_probability=0.0,
            xtc_threshold=0.0,
            xtc_special_tokens=[],
        )

        # kwargs ثابت برای هر بار فراخوانی generate
        self._gen_kwargs = {
            "max_tokens": max_tokens,
            "sampler": self._sampler,
            "kv_bits": kv_bits,
            "kv_group_size": kv_group_size,
            "quantized_kv_start": 0,
        }

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        اگر temperature یا max_tokens در سطح call تغییر کند،
        یک sampler موقت می‌سازیم — وگرنه از _gen_kwargs پیش‌ساخته استفاده می‌کنیم.
        """
        import asyncio

        # phi-4 chat template
        prompt = (
            f"<|system|>\n{system}<|end|>\n"
            f"<|user|>\n{user}<|end|>\n"
            f"<|assistant|>\n"
        )

        # اگر پارامتر override شد، kwargs جدید بساز
        if temperature is not None or max_tokens is not None:
            from mlx_lm.sample_utils import make_sampler as _ms
            gen_kwargs = {
                **self._gen_kwargs,
                **({"max_tokens": max_tokens} if max_tokens else {}),
                **(
                    {
                        "sampler": _ms(
                            temp=temperature,
                            top_p=1.0,
                            min_p=0.0,
                            min_tokens_to_keep=1,
                            top_k=0,
                            xtc_probability=0.0,
                            xtc_threshold=0.0,
                            xtc_special_tokens=[],
                        )
                    }
                    if temperature is not None
                    else {}
                ),
            }
        else:
            gen_kwargs = self._gen_kwargs

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                verbose=False,
                **gen_kwargs,
            ),
        )
        return result

    async def close(self) -> None:
        pass  # MLX مدیریت حافظه خودکار دارد


# ─────────────────────────────────────────────
# Backend Factory
# ─────────────────────────────────────────────

def build_backend(cfg: dict) -> LLMBackend:
    """
    config["phi4_mini"]["backend"] می‌تواند باشد:
      "ollama"    → OllamaBackend
      "llama_cpp" → LlamaCppBackend
      "mlx"       → MLXBackend
    """
    backend_type = cfg.get("backend", "ollama")

    if backend_type == "ollama":
        return OllamaBackend(
            base_url=cfg["base_url"],
            model=cfg["model"],
            timeout=cfg.get("timeout", 30),
        )

    if backend_type == "llama_cpp":
        return LlamaCppBackend(
            model_path=cfg["model_path"],
            n_ctx=cfg.get("n_ctx", 4096),
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
            server_url=cfg.get("server_url"),  # اختیاری
        )

    if backend_type == "mlx":
        return MLXBackend(
            model_path=cfg["model_path"],
           # max_kv_size=cfg.get("max_kv_size", 4096),
        )

    raise ValueError(
        f"Unknown backend: {backend_type!r}. Choose: ollama | llama_cpp | mlx"
    )


# ─────────────────────────────────────────────
# Shared Utilities
# ─────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    JSON را از متن خام استخراج می‌کند.
    مدل‌ها گاهی markdown fence یا متن اضافه تولید می‌کنند.
    """
    # تلاش مستقیم
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # پیدا کردن اولین { ... } معتبر
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model output:\n{text[:300]}")


def _validate_understand(data: dict, query: str) -> dict:
    """
    اطمینان از اینکه خروجی understand() schema درستی دارد.
    مقادیر پیش‌فرض برای فیلدهای گمشده.
    """
    valid_intents = {
        "factual",
        "analytical",
        "comparative",
        "exploratory",
        "procedural",
    }
    known_tools = {t["name"] for t in TOOL_REGISTRY}

    data.setdefault("intent", "factual")
    data.setdefault("complexity", 1)
    data.setdefault("language", "en")
    data.setdefault("entities", [])
    data.setdefault("sub_questions", [query])
    data.setdefault("search_keywords", [query])
    data.setdefault("requires_realtime", False)
    data.setdefault("tool_calls", [])

    if data["intent"] not in valid_intents:
        data["intent"] = "factual"

    # فیلتر tool_calls — فقط ابزارهای موجود در registry
    data["tool_calls"] = [
        tc
        for tc in data["tool_calls"]
        if isinstance(tc, dict) and tc.get("tool") in known_tools
    ]

    # sub_questions نباید خالی باشد
    if not data["sub_questions"]:
        data["sub_questions"] = [query]

    return data


# ─────────────────────────────────────────────
# Main Client
# ─────────────────────────────────────────────

class Phi4MiniClient:
    """
    phi4-mini — Query Understanding + Tool-Calling + Planning + Gap Detection

    پشتیبانی از سه backend:
    - ollama     : اجرای محلی از طریق Ollama REST API
    - llama_cpp  : اجرای مستقیم GGUF (in-process یا server)
    - mlx        : اجرای بهینه روی Apple Silicon با mlx_lm

    نمونه config:
    {
        "phi4_mini": {
            "backend": "mlx",          # یا "ollama" یا "llama_cpp"
            "model_path": "/models/phi-4-mini",   # برای mlx و llama_cpp
            "base_url": "http://localhost:11434",  # برای ollama
            "model": "phi4-mini",                 # برای ollama
            "timeout": 30
        }
    }
    """
    _DEFAULT_CONFIG = {
        "phi3_mini": {
            "backend": "mlx",
            "model_path": "/Users/dbk/Desktop/RAG/models/phi3_mini",
        }
    }

    defconfig:dict = { 
        "phi3_mini": {
                "backend":"mlx",
                "model_path": "/Users/dbk/Desktop/RAG/models/phi3_mini",
                 }
    }
                

    def __init__(self,config: dict | None = None ):
        print(config)
        print(self.defconfig)
 
        resolved = config or self._DEFAULT_CONFIG
        cfg = resolved.get("phi3_mini") or self._DEFAULT_CONFIG["phi3_mini"]
        print(cfg)

      
        self._backend: LLMBackend = build_backend(cfg)
        self._tools_desc: str = self._build_tools_desc()

    def _build_tools_desc(self) -> str:
        """یک‌بار ساخته می‌شود — TOOL_REGISTRY تغییر نمی‌کند"""
        return json.dumps(
            [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                }
                for t in TOOL_REGISTRY
            ],
            ensure_ascii=False,
            indent=2,
        )
        

    # ── Public API ────────────────────────────

    async def understand(self, query: str, history: list[dict]) -> dict:
        """
        مرحله UNDERSTAND در Orchestrator.

        خروجی:
        intent, complexity, language, entities,
        sub_questions, tool_calls (validated), search_keywords, requires_realtime
        """
        
        print(f"phi3 query : undestanding :  start->>   {query}   <<-end")
        print(f"phi3 history : undestanding :  start->>   {history}   <<-end")
        print(f"phi3 tools : undestanding :  start->>   {self._tools_desc}   <<-end")

        prompt = UNDERSTAND_TEMPLATE.format(
            query=query,
            history=json.dumps(history[-3:],
            ensure_ascii=False),
            tools=self._tools_desc,
        )
        print(f"phi3 json propt : undestanding :  start->>   {prompt}   <<-end")
        raw = await self._backend.chat(system=UNDERSTAND_SYSTEM, user=prompt)
        print(f"phi3 json answer: undestanding :  start->>   {raw}   <<-endAnswer")
        try:
            data = _extract_json(raw)
            return _validate_understand(data, query)
        except Exception as e:
            logger.warning(f"understand() parse failed: {e}")
            return self._fallback_understand(query)

    async def plan(self, query: str, understanding: dict) -> dict:
        """
        مرحله PLAN — تولید گراف اجرایی برای Orchestrator.

        خروجی:
        steps[{id, action, description, depends_on, tool, args}],
        estimated_hops, strategy
        """
        prompt = PLAN_TEMPLATE.format(
            query=query,
            understanding=json.dumps(understanding, ensure_ascii=False, indent=2),
        )
        raw = await self._backend.chat(system=PLAN_SYSTEM, user=prompt)
        print(f"phi3 json: Plan :  start->>   {raw}   <<-end")

        try:
            data = _extract_json(raw)
            return self._validate_plan(data, understanding)
        except Exception as e:
            logger.warning(f"plan() parse failed: {e}")
            return self._fallback_plan(understanding)

    async def detect_gaps(self, query: str, context: str) -> dict:
        """
        مرحله ASSESS — آیا context کافی است؟

        خروجی:
        sufficient, confidence, missing_aspects,
        search_queries, suggested_sources
        """
        prompt = GAP_TEMPLATE.format(query=query, context=context[:3000])
        raw = await self._backend.chat(system=GAP_SYSTEM, user=prompt)

        try:
            data = _extract_json(raw)
            data.setdefault("sufficient", False)
            data.setdefault("confidence", 0.0)
            data.setdefault("missing_aspects", [])
            data.setdefault("search_queries", [query])
            data.setdefault("suggested_sources", ["vector"])
            return data
        except Exception as e:
            logger.warning(f"detect_gaps() parse failed: {e}")
            return {
                "sufficient": False,
                "confidence": 0.0,
                "missing_aspects": [],
                "search_queries": [query],
                "suggested_sources": ["vector"],
            }

    async def select_tools(self, query: str, context: str = "") -> list[dict]:
        """
        Tool selection مستقل — برای مواقعی که Orchestrator
        فقط به tool-call نیاز دارد بدون full understanding.
        """
        understanding = await self.understand(query, [])
        return understanding.get("tool_calls", [])

    async def close(self) -> None:
        await self._backend.close()

    # ── Validation Helpers ────────────────────

    def _validate_plan(self, data: dict, understanding: dict) -> dict:
        valid_actions = {
            "retrieve",
            "reason",
            "search",
            "summarize",
            "compare",
            "validate",
        }
        known_tools = {t["name"] for t in TOOL_REGISTRY}

        data.setdefault("steps", [])
        data.setdefault("estimated_hops", 1)
        data.setdefault("strategy", "sequential")

        validated_steps = []
        for step in data["steps"]:
            if not isinstance(step, dict):
                continue
            step.setdefault("id", len(validated_steps) + 1)
            step.setdefault("action", "retrieve")
            step.setdefault("description", "")
            step.setdefault("depends_on", [])
            step.setdefault("tool", None)
            step.setdefault("args", {})

            if step["action"] not in valid_actions:
                step["action"] = "retrieve"
            if step["tool"] and step["tool"] not in known_tools:
                step["tool"] = None

            validated_steps.append(step)

        # اگر plan خالی بود، از sub_questions بساز
        if not validated_steps:
            for i, sq in enumerate(understanding.get("sub_questions", []), 1):
                validated_steps.append(
                    {
                        "id": i,
                        "action": "retrieve",
                        "description": sq,
                        "depends_on": [],
                        "tool": None,
                        "args": {},
                    }
                )

        data["steps"] = validated_steps
        return data

    # ── Fallbacks ─────────────────────────────

    @staticmethod
    def _fallback_understand(query: str) -> dict:
        return {
            "intent": "factual",
            "complexity": 1,
            "language": "en",
            "entities": [],
            "sub_questions": [query],
            "tool_calls": [],
            "search_keywords": [query],
            "requires_realtime": False,
        }

    @staticmethod
    def _fallback_plan(understanding: dict) -> dict:
        steps = [
            {
                "id": i + 1,
                "action": "retrieve",
                "description": sq,
                "depends_on": [],
                "tool": None,
                "args": {},
            }
            for i, sq in enumerate(understanding.get("sub_questions", []))
        ]
        return {
            "steps": steps
            or [
                {
                    "id": 1,
                    "action": "retrieve",
                    "description": "retrieve context",
                    "depends_on": [],
                    "tool": None,
                    "args": {},
                }
            ],
            "estimated_hops": len(steps) or 1,
            "strategy": "sequential",
        }
# import json
# import httpx
# from llm.tool_registry import TOOL_REGISTRY


# UNDERSTAND_SYSTEM = """You are a query understanding engine.
# Your job:
# 1. Detect intent (factual / analytical / comparative / exploratory)
# 2. Extract named entities with types
# 3. Decompose complex queries into atomic sub-questions
# 4. Select appropriate tools from the registry

# Always respond in valid JSON.
# """

# UNDERSTAND_TEMPLATE = """
# Query: {query}
# Conversation history: {history}
# Available tools: {tools}

# Respond with JSON:
# {{
#   "intent": "...",
#   "entities": [{{"text":"...","type":"PERSON|ORG|LOC|CONCEPT|EVENT"}}],
#   "sub_questions": ["..."],
#   "tool_calls": [{{"tool":"...","args":{{}}}}],
#   "search_keywords": ["..."]
# }}
# """


# class Phi4MiniClient:
#     """
#     phi4-mini  →  Tool-Call + Query Understanding
#     اجرا روی Ollama یا vLLM
#     """

#     def __init__(self, config: dict):
#         self.base_url  = config["phi4_mini"]["base_url"]
#         self.model     = config["phi4_mini"]["model"]   # "phi4-mini"
#         self.timeout   = config["phi4_mini"].get("timeout", 30)
#         self._client   = httpx.AsyncClient(timeout=self.timeout)

#     async def understand(
#         self,
#         query:   str,
#         history: list[dict],
#     ) -> dict:
#         """
#         phi4-mini را برای فهم query و tool selection صدا می‌زند
#         """
#         tools_desc = json.dumps(
#             [{
#                 "name": t["name"],
#                 "description": t["description"],
#                 "parameters": t["parameters"],
#             } for t in TOOL_REGISTRY],
#             ensure_ascii=False,
#         )

#         prompt = UNDERSTAND_TEMPLATE.format(
#             query=query,
#             history=json.dumps(history[-3:], ensure_ascii=False),
#             tools=tools_desc,
#         )

#         response = await self._call_ollama(
#             system=UNDERSTAND_SYSTEM,
#             user=prompt,
#         )

#         try:
#             return json.loads(response)
#         except json.JSONDecodeError:
#             # fallback محافظ
#             return {
#                 "intent": "factual",
#                 "entities": [],
#                 "sub_questions": [query],
#                 "tool_calls": [],
#                 "search_keywords": [query],
#             }

#     async def detect_gaps(
#         self, query: str, context: str
#     ) -> dict:
#         """تشخیص خلأ دانشی"""
#         prompt = f"""
# Query: {query}
# Available context:
# {context}

# Is the context sufficient to answer the query?
# Respond in JSON:
# {{
#   "sufficient": true/false,
#   "confidence": 0.0-1.0,
#   "missing_aspects": ["..."],
#   "search_queries": ["..."]
# }}
# """
#         response = await self._call_ollama(
#             system="You are a knowledge gap detector. Respond only in JSON.",
#             user=prompt,
#         )
#         try:
#             return json.loads(response)
#         except json.JSONDecodeError:
#             return {"sufficient": False, "confidence": 0.0,
#                     "missing_aspects": [], "search_queries": [query]}

#     async def _call_ollama(self, system: str, user: str) -> str:
#         payload = {
#             "model":  self.model,
#             "stream": False,
#             "messages": [
#                 {"role": "system",  "content": system},
#                 {"role": "user",    "content": user},
#             ],
#             "options": {
#                 "temperature": 0.0,
#                 "num_predict": 1024,
#             },
#         }
#         resp = await self._client.post(
#             f"{self.base_url}/api/chat",
#             json=payload,
#         )
#         resp.raise_for_status()
#         return resp.json()["message"]["content"]
