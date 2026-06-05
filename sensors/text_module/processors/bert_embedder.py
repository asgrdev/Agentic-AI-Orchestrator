from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import time
from typing import Any
import warnings
import logging

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None

try:
    from runtime_module_io_logger import log_module_io
except Exception:  # pragma: no cover
    def log_module_io(*args: Any, **kwargs: Any) -> None:
        return

@dataclass
class BertEmbedderConfig:
    model_name_or_path: str = "bert-base-cased"
    device: str = "cpu"
    max_tokens: int = 256
    model_name: str = "bert_base_cased_cls"

class BertEmbedder:
    def __init__(self, cfg: BertEmbedderConfig) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for BertEmbedder.")
        if AutoTokenizer is None or AutoModel is None:
            raise RuntimeError("transformers is required. Install: pip install transformers")
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        tok_kwargs = {}
        tok_logger = logging.getLogger("transformers.tokenization_utils_base")
        prev_tok_level = int(tok_logger.level)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*incorrect regex pattern.*fix_mistral_regex.*",
                category=Warning,
            )
            tok_logger.setLevel(logging.ERROR)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, fix_mistral_regex=True, **tok_kwargs)
            except TypeError:
                self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, **tok_kwargs)
            finally:
                tok_logger.setLevel(prev_tok_level)
        self.model = AutoModel.from_pretrained(cfg.model_name_or_path)
        # Prevent silently running with broken local checkpoints.
        try:
            tok_vocab = int(len(self.tokenizer))
            emb_vocab = int(self.model.get_input_embeddings().weight.shape[0])
            if tok_vocab != emb_vocab:
                raise RuntimeError(
                    f"Tokenizer/model vocab mismatch for {cfg.model_name_or_path}: "
                    f"tokenizer={tok_vocab}, model_embeddings={emb_vocab}"
                )
        except Exception:
            raise
        self.model.to(self.device).eval()
        log_module_io(
            module="bert.embedder",
            stage="model_load",
            inputs={
                "model_name_or_path": cfg.model_name_or_path,
                "device": cfg.device,
                "max_tokens": int(cfg.max_tokens),
            },
            outputs={
                "tokenizer_vocab_size": int(len(self.tokenizer)),
                "embedding_vocab_size": int(self.model.get_input_embeddings().weight.shape[0]),
                "hidden_size": int(self.model.config.hidden_size) if hasattr(self.model, "config") else None,
            },
        )

    def embed(self, text: str) -> np.ndarray:
        import torch
        t0 = time.time()
        text_s = str(text or "")
        log_module_io(
            module="bert.embedder",
            stage="input",
            inputs={
                "text": text_s,
                "char_count": len(text_s),
                "word_count": len([w for w in text_s.split() if w]),
                "max_tokens": int(self.cfg.max_tokens),
            },
        )
        try:
            inputs = self.tokenizer(
                text_s,
                return_tensors="pt",
                truncation=True,
                max_length=int(self.cfg.max_tokens),
                padding=False,
            )
            token_count = int(inputs.get("input_ids").shape[-1]) if "input_ids" in inputs else 0
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model(**inputs)
                cls = out.last_hidden_state[:, 0, :]
                vec = cls.squeeze(0).detach().float().cpu().numpy()
            norm = float(np.linalg.norm(vec) + 1e-12)
            out_vec = (vec / norm).astype("float32")
            log_module_io(
                module="bert.embedder",
                stage="output",
                outputs={
                    "token_count": token_count,
                    "vector_dim": int(out_vec.shape[0]) if out_vec.ndim == 1 else None,
                    "vector_l2_norm": float(np.linalg.norm(out_vec) + 1e-12),
                    "vector_head": [float(x) for x in out_vec[:8]],
                    "latency_ms": round((time.time() - t0) * 1000.0, 3),
                },
            )
            return out_vec
        except Exception as exc:
            log_module_io(
                module="bert.embedder",
                stage="error",
                error=repr(exc),
                metadata={"latency_ms": round((time.time() - t0) * 1000.0, 3)},
            )
            raise
