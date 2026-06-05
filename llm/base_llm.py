import httpx


ANSWER_SYSTEM = """You are a precise, factual AI assistant.
You reason step by step using the provided knowledge graph context.
Always cite your sources. If uncertain, say so explicitly.
"""

ANSWER_TEMPLATE = """
## Knowledge Graph Context
{graph_context}

## Retrieved Documents
{doc_context}

## Reasoning Trace So Far
{reasoning}

## Question
{query}

## Instructions
1. Synthesize the context above
2. Reason step by step
3. Provide a direct, well-cited answer
4. Indicate confidence: HIGH / MEDIUM / LOW
"""


class GraniteClient:
    """
    granite4-7b → Deep reasoning + Final answer generation
    اجرا روی Ollama یا vLLM
    """

    def __init__(self, config: dict):
        self.base_url = config["granite"]["base_url"]
        self.model    = config["granite"]["model"]   # "granite4-7b"
        self.params   = config["granite"].get("params", {
            "temperature": 0.2,
            "num_predict": 2048,
            "top_p": 0.9,
        })
        self._client  = httpx.AsyncClient(timeout=120)

    async def answer(
        self,
        query:         str,
        graph_context: str,
        doc_context:   str,
        reasoning:     str = "",
    ) -> dict:
        prompt = ANSWER_TEMPLATE.format(
            query=query,
            graph_context=graph_context,
            doc_context=doc_context,
            reasoning=reasoning,
        )

        response = await self._call(ANSWER_SYSTEM, prompt)

        return {
            "answer":     response,
            "confidence": self._extract_confidence(response),
        }

    async def reason_step(
        self,
        query:       str,
        context:     str,
        prev_steps:  list[str],
    ) -> str:
        """یک قدم chain-of-thought"""
        steps_text = "\n".join(
            f"Step {i+1}: {s}" for i, s in enumerate(prev_steps)
        )
        prompt = f"""
Context: {context}
Question: {query}
Previous reasoning steps:
{steps_text}

What is the next logical reasoning step?
Be concise and specific.
"""
        return await self._call(
            "You are a step-by-step reasoner. One step at a time.",
            prompt,
        )

    async def _call(self, system: str, user: str) -> str:
        payload = {
            "model":   self.model,
            "stream":  False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "options": self.params,
        }
        resp = await self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _extract_confidence(self, text: str) -> float:
        text_lower = text.lower()
        if "confidence: high"   in text_lower:
            return 0.90
        if "confidence: medium" in text_lower:
            return 0.65
        if "confidence: low"    in text_lower:
            return 0.35
        return 0.50
