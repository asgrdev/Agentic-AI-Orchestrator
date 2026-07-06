"""
کاتالوگ مدل‌های لوکال — منبع واحد برای «چه مدلی داریم و به چه دردی می‌خورد».

هر ورودی: نام پوشه/فایل در `models/`، نوع، قابلیت‌ها، و skill هایی که از آن
استفاده می‌کنند. planner و سناریوهای چند-عاملی (A2A) از طریق skill
«list_local_models» به این کاتالوگ دسترسی دارند تا بدانند برای هر مرحله‌ی
plan چه ابزاری در دسترس است.

مسیر هر مدل با core.model_paths.resolve_model_path حل می‌شود:
اول `models/<name>` لوکال، بعد fallback.
"""
from __future__ import annotations

from core.model_paths import resolve_model_path

# ── تعریف ثابت کاتالوگ ───────────────────────────────────────────────
_CATALOG: list[dict] = [
    # ── LLM ها ──
    {
        "name": "phi3_mini", "kind": "llm", "backend": "mlx",
        "capabilities": ["query understanding", "planning", "tool-calling"],
        "used_by": ["understand_and_plan (phi4_mini_factory)"],
        "fallback": "/Users/dbk/Desktop/RAG/models/phi3_mini",
    },
    {
        "name": "granite4_3b", "kind": "llm", "backend": "mlx",
        "capabilities": ["reasoning", "answer generation", "chat"],
        "used_by": ["reason step (llm_client_factory)", "direct_answer"],
        "fallback": "/Users/dbk/Desktop/RAG/models/granite4_3b",
    },
    {
        "name": "granite4-7b", "kind": "llm", "backend": "mlx",
        "capabilities": ["reasoning (higher quality, ~15GB Metal)"],
        "used_by": ["اختیاری: GRANITE_MODEL_PATH"],
        "fallback": "/Users/dbk/Desktop/RAG/models/granite4-7b",
    },
    {
        "name": "Phi-3-mini-4k-instruct-q4-2.gguf", "kind": "llm", "backend": "llama_cpp",
        "capabilities": ["CPU-only generation (GGUF)"],
        "used_by": ["اختیاری: backend=llama_cpp در Phi4MiniClient"],
        "fallback": "",
    },
    {
        "name": "gpt2-small", "kind": "llm", "backend": "hf",
        "capabilities": ["lightweight text generation / perplexity"],
        "used_by": [],
        "fallback": "",
    },
    # ── Embedding ها ──
    {
        "name": "Qwen3-Embedding-0.6B", "kind": "embedding", "backend": "torch",
        "capabilities": ["RAG retrieval embedding (dim=1024)"],
        "used_by": ["retrieve/ingest (Qwen3EmbeddingClient)"],
        "fallback": "/Users/dbk/Desktop/agentic-graph-RAG/models/Qwen3-Embedding-0.6B",
    },
    {
        "name": "all-mpnet-base-v2", "kind": "embedding", "backend": "torch",
        "capabilities": ["sentence similarity (سبک‌تر از Qwen3)"],
        "used_by": ["compute_vector_similarity", "cluster_vectors"],
        "fallback": "",
    },
    {
        "name": "bert-base-cased", "kind": "text-encoder", "backend": "torch",
        "capabilities": ["token embedding", "NER backbone"],
        "used_by": ["embed_text_with_bert", "extract_entities_bert"],
        "fallback": "",
    },
    # ── Vision (YOLO11) ──
    {
        "name": "yolo11x.pt", "kind": "vision", "backend": "ultralytics",
        "capabilities": ["object detection"],
        "used_by": ["detect_objects_in_image", "analyze_uploaded_file(image)"],
        "fallback": "yolov8n.pt",
    },
    {
        "name": "yolo11xpos.pt", "kind": "vision", "backend": "ultralytics",
        "capabilities": ["human pose estimation (keypoints)"],
        "used_by": ["estimate_pose_in_image"],
        "fallback": "",
    },
    {
        "name": "yolo11xseg.pt", "kind": "vision", "backend": "ultralytics",
        "capabilities": ["instance segmentation (masks)"],
        "used_by": ["segment_image_objects"],
        "fallback": "",
    },
    # ── Audio ──
    {
        "name": "whisper-medium", "kind": "audio-asr", "backend": "whisper",
        "capabilities": ["speech-to-text (multi-lingual)"],
        "used_by": ["transcribe_audio_file", "analyze_uploaded_file(audio)"],
        "fallback": "",
    },
    {
        "name": "wav2vec2_fairseq_base_ls960_asr_ls960.pth", "kind": "audio-asr",
        "backend": "torch",
        "capabilities": ["English ASR (جایگزین سبک whisper)"],
        "used_by": [],
        "fallback": "",
    },
    {
        "name": "Cnn14.pth", "kind": "audio-tagging", "backend": "torch",
        "capabilities": ["audio event detection (PANNs, 527 کلاس AudioSet)"],
        "used_by": ["detect_audio_events"],
        "fallback": "",
    },
    {
        "name": "Cnn10.pth", "kind": "audio-tagging", "backend": "torch",
        "capabilities": ["audio event detection (PANNs سبک‌تر)"],
        "used_by": [],
        "fallback": "",
    },
    {
        "name": "efficientat.pt", "kind": "audio-tagging", "backend": "torch",
        "capabilities": ["efficient audio tagging"],
        "used_by": ["detect_audio_events (EfficientATProcessor)"],
        "fallback": "",
    },
    # ── NLP ──
    {
        "name": "en_core_web_trf-any-py3-none-any.whl", "kind": "nlp", "backend": "spacy",
        "capabilities": ["NER / linguistic pipeline (transformer)"],
        "used_by": ["NERExtractor (spacy_model)"],
        "fallback": "",
    },
]


def get_model_catalog() -> list[dict]:
    """کاتالوگ با مسیر resolve شده و وضعیت موجود بودن روی دیسک"""
    from pathlib import Path

    out = []
    for entry in _CATALOG:
        e = dict(entry)
        path = resolve_model_path(e["name"], e.get("fallback") or None)
        e["path"] = path
        e["available"] = Path(path).exists()
        out.append(e)
    return out


def catalog_summary() -> str:
    """خلاصه‌ی متنی برای پرامپت planner / پاسخ skill"""
    lines = []
    for e in get_model_catalog():
        mark = "✅" if e["available"] else "❌"
        caps = ", ".join(e["capabilities"])
        lines.append(f"{mark} {e['name']} [{e['kind']}]: {caps}")
    return "\n".join(lines)
