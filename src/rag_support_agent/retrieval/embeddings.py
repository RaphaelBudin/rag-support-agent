"""Embedding providers behind a tiny interface.

Three implementations:
  - HashEmbedder: deterministic feature-hashing, no API key. Lets the whole repo run
    (ingest -> retrieve -> eval) with zero setup so a reviewer can try it in minutes.
    Not semantically strong — it's a dev/test stand-in, clearly labeled.
  - OpenAIEmbedder / GeminiEmbedder: the real, semantic ones.

Swap via EMBEDDING_PROVIDER. All must emit vectors of EMBEDDING_DIM so the pgvector
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


class GeminiEmbedder:
    """Real embeddings via Google Gemini (``google-genai``).

    ``gemini-embedding-001`` pre-normalizes only its full 3072-d output. When we request
    a truncated dimensionality (1536, to match the pgvector column) the vectors come back
    un-normalized (L2-norm ~0.7), so we L2-normalize here — otherwise cosine similarity
    against the stored, normalized vectors would be off.
    """

    _BATCH = 100  # gemini-embedding-001 caps inputs per request; batch for larger corpora.

    def __init__(self, model: str, dim: int, api_key: str | None) -> None:
        from google import genai

        self.dim = dim
        self.model = model
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _normalize(v: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        cfg = types.EmbedContentConfig(output_dimensionality=self.dim)
        out: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH):
            resp = self._client.models.embed_content(
                model=self.model, contents=texts[i : i + self._BATCH], config=cfg
            )
            out.extend(self._normalize(list(e.values)) for e in resp.embeddings)
        return out


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    if s.embedding_provider == "openai":
        return OpenAIEmbedder(s.embedding_model, s.embedding_dim, s.openai_api_key)
    if s.embedding_provider == "gemini":
        # Fall back to a Gemini model name if the config still holds an OpenAI default.
        model = s.embedding_model if s.embedding_model.startswith("gemini") else "gemini-embedding-001"
        return GeminiEmbedder(model, s.embedding_dim, s.gemini_api_key)
    if s.embedding_provider == "hash":
        return HashEmbedder(s.embedding_dim)
    raise ValueError(f"unknown embedding_provider: {s.embedding_provider!r}")
