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

UNDERSTAND_PLAN_SYSTEM = """\
You are a query understanding and planning engine for an AgenticGraphRAG system.
Respond ONLY with valid JSON, no markdown, no explanation.
"""

UNDERSTAND_PLAN_TEMPLATE = """\
Query: {query}
History (last 3): {history}
Tools: {tools}

{{
  "intent": "factual|analytical|comparative|exploratory|procedural",
  "complexity": 1,
  "language": "en|fa|...",
  "entities": [{{"text": "...", "type": "PERSON|ORG|LOC|CONCEPT|EVENT|DATE|PRODUCT", "relevance": 0.9}}],
  "sub_questions": ["..."],
  "tool_calls": [{{"tool": "...", "args": {{}}, "reason": "..."}}],
  "search_keywords": ["..."],
  "requires_realtime": false,
  "steps": [
    {{"id": 1, "action": "retrieve|reason|search|summarize|compare|validate", "description": "...", "depends_on": [], "tool": null}}
  ],
  "strategy": "sequential|parallel|iterative"
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
        n_gpu_layers: int = 10,
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
    async def close(self) -> None:
        if self._llm is not None:
            try:
                del self._llm
            except:
                pass
        import gc
        gc.collect()
    
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
        loop = asyncio.get_running_loop()
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

    # کش مدل به core.model_gate منتقل شد — همان‌جا serial/concurrent کنترل می‌شود

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

        self._generate = generate

        # لود از طریق ModelGate — کش + کنترل serial/concurrent متمرکز
        from core.model_gate import get_model_gate
        self.model, self.tokenizer = get_model_gate().acquire(
            key=f"mlx:{model_path}",
            kind="llm",
            loader=lambda: load(model_path),
        )
        self._model_path = model_path

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
        
    async def close(self) -> None:
        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:
                mx.metal.clear_cache()
        except Exception:
            pass
    
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

        loop = asyncio.get_running_loop()
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
            n_gpu_layers=cfg.get("n_gpu_layers", -1),
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
    Extract the first complete JSON object from raw model output.
    Uses raw_decode so trailing text / extra objects are ignored cleanly.
    """
    text = text.strip()

    # strip markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text).rstrip("`").strip()

    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start)
            return obj
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model output:\n-----\n{text[:300]}")


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
        resolved = config or self._DEFAULT_CONFIG
        cfg = resolved.get("phi3_mini") or self._DEFAULT_CONFIG["phi3_mini"]
        logger.debug("Phi4MiniClient config: %s", cfg)

      
        self._backend: LLMBackend = build_backend(cfg)
        self._tools_desc: str = self._build_tools_desc_withparam()
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()   
        
        
    def _build_tools_desc_withparam(self) -> str:
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
        
    def _build_tools_desc(self) -> str:
         """بدون parameters — مدل فقط باید tool انتخاب کنه، نه call بسازه"""
         return json.dumps(
             [{"name": t["name"],
               "description": t["description"]
               } 
              for t in TOOL_REGISTRY],
             ensure_ascii=False,
         )
         

    # ── Public API ────────────────────────────
    async def understand_and_plan(
        self,
        query: str,
        history: list[dict],
        max_retries: int = 2,
        timeout: float = 300.0
    ) -> dict:
        """
        Combined understanding and planning with enhanced reliability.
        
        Features:
        - Retry mechanism for transient failures
        - Timeout protection
        - Enhanced validation
        - Detailed logging
        - Quality scoring
        
        Args:
            query: User query
            history: Conversation history
            max_retries: Maximum retry attempts (default: 2)
            timeout: Timeout in seconds (default: 30.0)
            
        Returns:
            Combined understanding and plan dict with quality score
        """
        import asyncio
        from time import time
        
        start_time = time()
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"understand_and_plan attempt {attempt + 1}/{max_retries + 1} for query: {query[:50]}...")
                
                # Prepare prompt
                prompt = UNDERSTAND_PLAN_TEMPLATE.format(
                    query=query,
                    history=json.dumps(history[-3:], ensure_ascii=False),
                    tools=self._tools_desc,
                )
                
                # Call LLM with timeout
                raw = await asyncio.wait_for(
                    self._backend.chat(
                        system=UNDERSTAND_PLAN_SYSTEM,
                        user=prompt,
                        max_tokens=512,
                    ),
                    timeout=timeout
                )
                
                # Extract and validate
                data = _extract_json(raw)
                data = _validate_understand(data, query)
                data = self._validate_plan(data, data)
                
                # Add quality score
                quality_score = self._calculate_quality_score(data, query)
                data['quality_score'] = quality_score
                data['processing_time'] = time() - start_time
                data['attempt'] = attempt + 1
                
                # Log success
                logger.info(
                    f"understand_and_plan succeeded: "
                    f"quality={quality_score:.2f}, "
                    f"time={data['processing_time']:.2f}s, "
                    f"attempt={attempt + 1}"
                )
                
                # If quality is too low and we have retries left, try again
                if quality_score < 0.5 and attempt < max_retries:
                    logger.warning(f"Low quality score ({quality_score:.2f}), retrying...")
                    continue
                
                return data
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                logger.warning(f"understand_and_plan timeout on attempt {attempt + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {str(e)}"
                logger.warning(f"understand_and_plan JSON error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"understand_and_plan error on attempt {attempt + 1}: {e}", exc_info=True)
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
        
        # All retries failed - use fallback
        logger.error(f"understand_and_plan failed after {max_retries + 1} attempts: {last_error}")
        fb = self._fallback_understand(query)
        fb_plan = self._fallback_plan(fb)
        result = {**fb, **fb_plan}
        result['quality_score'] = 0.3  # Low score for fallback
        result['processing_time'] = time() - start_time
        result['attempt'] = max_retries + 1
        result['fallback'] = True
        result['error'] = last_error
        
        return result
 
 
    async def understand(self, query: str, history: list[dict]) -> dict:
        """
        مرحله UNDERSTAND در Orchestrator.

        خروجی:
        intent, complexity, language, entities,
        sub_questions, tool_calls (validated), search_keywords, requires_realtime
        """
        
        prompt = UNDERSTAND_TEMPLATE.format(
            query=query,
            history=json.dumps(history[-3:], ensure_ascii=False),
            tools=self._tools_desc,
        )
        raw = await self._backend.chat(system=UNDERSTAND_SYSTEM, user=prompt)
        try:
            data = _extract_json(raw)
            return _validate_understand(data, query)
        except Exception as e:
            logger.warning(f"understand() parse failed: {e}")
            return self._fallback_understand(query)
    def _calculate_quality_score(self, data: dict, query: str) -> float:
        """
        Calculate quality score for understanding and plan output.
        
        Score components:
        - Has entities: +0.2
        - Has sub_questions: +0.2
        - Has valid steps: +0.3
        - Steps have dependencies: +0.1
        - Complexity matches query: +0.2
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check entities
        if data.get('entities') and len(data['entities']) > 0:
            score += 0.2
        
        # Check sub_questions
        if data.get('sub_questions') and len(data['sub_questions']) > 0:
            score += 0.2
        
        # Check steps
        steps = data.get('steps', [])
        if steps and len(steps) > 0:
            score += 0.3
            
            # Check if steps have dependencies (indicates thoughtful planning)
            has_deps = any(step.get('depends_on') for step in steps)
            if has_deps:
                score += 0.1
        
        # Check complexity alignment
        query_len = len(query.split())
        expected_complexity = min(5, max(1, query_len // 5))
        actual_complexity = data.get('complexity', 1)
        
        if abs(expected_complexity - actual_complexity) <= 1:
            score += 0.2
        
        return min(1.0, score)


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
            # اگه context خیلی کوتاهه، مستقیم برگردون
        if len(context.strip()) < 100:
            return {
                "sufficient": False,
                "confidence": 0.0,
                "missing_aspects": ["no context retrieved"],
                "search_queries": [query],
                "suggested_sources": ["vector"],
            }
    
        # context رو کوتاه‌تر بفرست
        prompt = GAP_TEMPLATE.format(query=query, context=context[:1500])  # 3000 → 1500
        raw = await self._backend.chat(
            system=GAP_SYSTEM,
            user=prompt,
            max_tokens=256,  # gap detection نیاز به خروجی بلند نداره
        )
       

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

            # Tool must be compatible with the step action; e.g. a "retrieve"
            # step should not carry a "calculate" tool with no expression arg.
            _retrieval_tools = {None, "retrieve", "search", "graph_query", "web_search"}
            _compute_tools = {"calculate"}
            if step["action"] == "retrieve" and step["tool"] in _compute_tools:
                step["tool"] = None
                step["args"] = {}

            # Every retrieval/search step must carry its own refined query,
            # otherwise executors fall back to the raw user query and each
            # tool ends up receiving a different message for the same step.
            if (
                step["action"] in {"retrieve", "search"}
                or step["tool"] in {"retrieve", "search", "web_search"}
            ):
                if not step["args"].get("query") and step["description"]:
                    step["args"]["query"] = step["description"]

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
                        "args": {"query": sq},
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
                "args": {"query": sq},
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
