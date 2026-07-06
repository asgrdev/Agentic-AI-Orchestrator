from typing import Callable, Any

# List format — each entry: {name, description, parameters}
# prompt_understanding.py iterates this as a list[dict] to build the tool prompt
# and to validate tool_calls returned by the LLM.
TOOL_REGISTRY: list[dict] = []

# Fast name→entry lookup used by SkillExecutor
_REGISTRY_MAP: dict[str, dict] = {}


def register_tool(name: str, description: str = "", parameters: dict | None = None):
    """Decorator that registers a callable into both the list and the lookup map."""
    def decorator(fn: Callable) -> Callable:
        entry = {
            "name": name,
            "description": description,
            "parameters": parameters or {},
            "fn": fn,
        }
        # avoid duplicate registrations on module reload
        if name not in _REGISTRY_MAP:
            TOOL_REGISTRY.append(entry)
            _REGISTRY_MAP[name] = entry
        return fn
    return decorator


def get_tool(name: str) -> dict | None:
    return _REGISTRY_MAP.get(name)


# ── Default tools — these mirror SkillRegistry._register_default_skills() ──

@register_tool(
    "search",
    "Search for information across vector store and knowledge graph",
    {"query": "string", "sources": "list[string] — optional, e.g. ['vector','graph']"},
)
async def _search(query: str, sources: list[str] | None = None) -> dict:
    return {"query": query, "sources": sources or ["vector", "graph"], "results": []}


@register_tool(
    "web_search",
    "Search the web for real-time or external information",
    {"query": "string", "num_results": "int — optional, default 5"},
)
async def _web_search(query: str, num_results: int = 5) -> dict:
    return {"query": query, "num_results": num_results, "results": []}


@register_tool(
    "graph_query",
    "Run a Cypher query against the knowledge graph",
    {"cypher": "string — valid Cypher query"},
)
async def _graph_query(cypher: str) -> dict:
    return {"cypher": cypher, "results": []}


@register_tool(
    "calculate",
    "Evaluate a mathematical expression and return the result",
    {"expression": "string — e.g. '2 * (3 + 4)'"},
)
async def _calculate(expression: str) -> dict:
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


@register_tool(
    "summarize",
    "Summarize a block of text into a concise paragraph",
    {"text": "string", "max_sentences": "int — optional, default 3"},
)
async def _summarize(text: str, max_sentences: int = 3) -> dict:
    return {"text": text[:200], "max_sentences": max_sentences, "summary": ""}


@register_tool(
    "retrieve",
    "Retrieve relevant document chunks from the vector store for a query",
    {"query": "string", "top_k": "int — optional, default 5"},
)
async def _retrieve(query: str, top_k: int = 5) -> dict:
    return {"query": query, "top_k": top_k, "chunks": []}


# ── Media / local-model tools ────────────────────────────────────────
# اجرای واقعی این‌ها از طریق global skill registry انجام می‌شود
# (agents/sensor_skills.py) — این ورودی‌ها منوی planner هستند تا در
# planهای چندمرحله‌ای و سناریوهای A2A قابل انتخاب باشند.

@register_tool(
    "detect_objects_in_image",
    "Detect objects in an image file with local YOLO11 "
    "(use only when the query references an image file path)",
    {"image_path": "string — path to the image", "conf_threshold": "float — optional"},
)
def _detect_objects_in_image(image_path: str, conf_threshold: float = 0.25) -> dict:
    from agents.sensor_skills import detect_objects_in_image
    return detect_objects_in_image(image_path, conf_threshold)


@register_tool(
    "estimate_pose_in_image",
    "Estimate human poses (keypoints) in an image with local YOLO11-pose",
    {"image_path": "string", "conf_threshold": "float — optional"},
)
def _estimate_pose_in_image(image_path: str, conf_threshold: float = 0.25) -> dict:
    from agents.sensor_skills import estimate_pose_in_image
    return estimate_pose_in_image(image_path, conf_threshold)


@register_tool(
    "segment_image_objects",
    "Instance-segment objects in an image with local YOLO11-seg (mask area per object)",
    {"image_path": "string", "conf_threshold": "float — optional"},
)
def _segment_image_objects(image_path: str, conf_threshold: float = 0.25) -> dict:
    from agents.sensor_skills import segment_image_objects
    return segment_image_objects(image_path, conf_threshold)


@register_tool(
    "transcribe_audio_file",
    "Transcribe speech in an audio file to text with local Whisper",
    {"audio_path": "string", "language": "string — optional, auto-detect if omitted"},
)
def _transcribe_audio_file(audio_path: str, language: str | None = None) -> dict:
    from agents.sensor_skills import transcribe_audio_file
    return transcribe_audio_file(audio_path, language)


@register_tool(
    "list_local_models",
    "List locally available AI models and their capabilities "
    "(useful to decide which tools can run in a multi-step plan)",
    {},
)
def _list_local_models() -> dict:
    from agents.sensor_skills import list_local_models
    return list_local_models()
