"""Embedding providers behind a tiny interface.

Two implementations:
  - HashEmbedder: deterministic feature-hashing, no API key. Lets the whole repo run
    (ingest -> retrieve -> eval) with zero setup so a reviewer can try it in minutes.
    Not semantically strong — it's a dev/test stand-in, clearly labeled.
  - OpenAIEmbedder: the real one, used in production.

Swap via EMBEDDING_PROVIDER. Both must emit vectors of EMBEDDING_DIM so the pgvector
column and any stored data stay compatible.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from rag_support_agent.config import Settings, get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic, keyless feature-hashing embedder (dev/test only)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    """Real embeddings via the OpenAI API."""

    def __init__(self, model: str, dim: int, api_key: str | None) -> None:
        from openai import OpenAI

        self.dim = dim
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    if s.embedding_provider == "openai":
        return OpenAIEmbedder(s.embedding_model, s.embedding_dim, s.openai_api_key)
    if s.embedding_provider == "hash":
        return HashEmbedder(s.embedding_dim)
    raise ValueError(f"unknown embedding_provider: {s.embedding_provider!r}")
