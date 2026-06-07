from __future__ import annotations

import os
import sys


def make_embedder(config):
    """
    Return an embedder object with an .embed(texts) -> ndarray method,
    or None if embedding is unavailable.

    Detection order:
      1. DOC_SEARCH_EMBEDDER env var
      2. CUDA available -> sentence-transformers
      3. Default -> fastembed
    """
    from doc_search_mcp.config import Config

    cfg: Config = config
    backend = cfg.embeddings.backend
    model = cfg.embeddings.model

    if backend == "auto":
        backend = _detect_backend()

    if backend == "fastembed":
        return _make_fastembed(model)
    if backend == "sentence-transformers":
        return _make_sentence_transformers(model)
    if backend == "ollama":
        return _make_ollama(model, cfg.embeddings.ollama_url)
    if backend == "none":
        return None

    raise ValueError(f"Unknown embeddings backend: {backend!r}")


def _detect_backend() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "sentence-transformers"
    except ImportError:
        pass
    return "fastembed"


def _make_fastembed(model: str):
    try:
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=model)
    except Exception as exc:
        print(f"[doc-search] fastembed unavailable: {exc}", file=sys.stderr)
        return None


def _make_sentence_transformers(model: str):
    try:
        from sentence_transformers import SentenceTransformer

        class _STEmbedder:
            def __init__(self, m):
                self._model = SentenceTransformer(m)

            def embed(self, texts):
                return self._model.encode(texts, normalize_embeddings=True)

        return _STEmbedder(model)
    except Exception as exc:
        print(f"[doc-search] sentence-transformers unavailable: {exc}", file=sys.stderr)
        return None


def _make_ollama(model: str, url: str):
    import urllib.request
    import json
    import numpy as np

    class _OllamaEmbedder:
        def __init__(self, model: str, url: str) -> None:
            self._model = model
            self._url = url.rstrip("/") + "/api/embeddings"

        def embed(self, texts: list[str]):
            vectors = []
            for text in texts:
                payload = json.dumps({"model": self._model, "prompt": text}).encode()
                req = urllib.request.Request(
                    self._url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Ollama embedding failed: HTTP {resp.status}")
                    data = json.loads(resp.read())
                vectors.append(data["embedding"])
            return np.array(vectors, dtype=np.float32)

    return _OllamaEmbedder(model, url)
