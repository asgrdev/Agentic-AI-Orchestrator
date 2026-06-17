# config.py
from __future__ import annotations
from typing import Callable, Any

def _make_config() -> dict:
    # ── فقط reference به کلاس‌ها، نه ساخت instance ───────────────

    def make_phi4_client():
        from llm.prompt_understanding import Phi4MiniClient
        return Phi4MiniClient()

    def make_granite_client():
        from llm.granite_client.mlx_client import MlxGraniteClient
        return MlxGraniteClient()

    def make_embed_client():
        from ingestion.embedding_generator import Qwen3EmbeddingClient
        return Qwen3EmbeddingClient(model_path="/Users/dbk/Desktop/agentic-graph-RAG/models/Qwen3-Embedding-0.6B")

    def make_context_builder():
        from ingestion.context_builder import ContextBuilder
        return ContextBuilder()

    return {
        # ── Lazy factories ─────────────────────────────────────────
        "llm_client_factory":        make_granite_client,
        "phi4_mini_factory":         make_phi4_client,
        "embed_client_factory":      make_embed_client,
        "context_builder_factory":   make_context_builder,

        # ── بقیه configها بدون تغییر ─────────────────────────────
        "kuzu_path": "./data/kuzu_db",
        "collection_name": "Chunks",
        "confidence_threshold": 0.65,

        "vector_top_k": 10,
        "hybrid_alpha": 0.7,
        "mmr_lambda": 0.6,

        "validator": {
            "score_threshold_answer": 0.75,
            "score_threshold_retrieve": 0.50,
            "min_answer_length": 10,
        },
        
        
          # ── KuzuDB (graph store) ─────────────────────────────────────
        "kuzu_path":            "./data/kuzu_db",

        # ── Weaviate (vector store) ──────────────────────────────────
        "weaviate": {
            "mode":     "docker",   # "docker" | "embedded" | "cloud"
            "host":     "localhost",
            "port":     8080,
            "grpc_port": 50051,
        },
        "collection_name":      "Chunks",

        # ── Orchestrator ─────────────────────────────────────────────
        "confidence_threshold": 0.65,

        # ── RetrieverAgent ───────────────────────────────────────────
        "vector_top_k":         10,
        "hybrid_alpha":         0.7,    # 1.0 = pure vector, 0.0 = pure BM25
        "mmr_lambda":           0.6,    # تنوع در MMR reranking
        "boost_factor":         0.3,    # graph-score boost
        "graph_hops":           2,
        "confidence_top_k":     3,

        # ── ValidatorAgent ───────────────────────────────────────────
        "validator": {
            "score_threshold_answer":   0.75,   # >= این → پاسخ نهایی
            "score_threshold_retrieve": 0.50,   # >= این → retrieve مجدد
            "min_answer_length":        10,     # < این → رد می‌شه
        },

        # ── ReasonerAgent ────────────────────────────────────────────
        "reasoner": {
            # اگر ReasonerAgent زیرکلیدهای اضافه داشت اینجا اضافه کن
        },

        # ── KnowledgeRefreshAgent ────────────────────────────────────
        "chunk_size":           400,
        "refresh_interval":     60,     # ثانیه
        "max_concurrent":       5,

        # ── DoclingProcessor ─────────────────────────────────────────
        "docling_ocr":          True,
        "chunk_overlap":        64,

        # ── NERExtractor ─────────────────────────────────────────────
        "spacy_model":          "en_core_web_trf" ,# "en-core-web-lg",
        "gliner_model":         "urchade/gliner_medium-v2.1",

        # ── LLMRelationExtractor ─────────────────────────────────────
        "relation_model":       "/Users/dbk/Desktop/agentic-graph-RAG/models/Qwen3-Embedding-0.6B",#None,   # اگر مدل جداگانه داری اینجا بذار

        # ── EntityExtractionPipeline ─────────────────────────────────
        "max_workers":          4,
    }

CONFIG = _make_config()
