"""Central configuration, loaded from environment / .env.

Kept in one place so provider choices (which embedder, which LLM) and behavioral
knobs (abstention threshold, top-k) are explicit and swappable without touching code.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # Embeddings. Provider "hash" is a deterministic, keyless dev/test embedder so the
    # repo runs end-to-end with no API key; "openai" and "gemini" are the real ones.
    embedding_provider: str = "hash"  # "hash" | "openai" | "gemini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # must match the provider's output and the DB column

    # Generation (M3). Provider "extractive" is a keyless default that builds the answer
    # from retrieved passages verbatim (grounding by construction, no API key) so the repo
    # runs end-to-end for a stranger; "gemini" is the real synthesizer.
    llm_provider: str = "extractive"  # "extractive" (keyless) | "gemini"
    generation_model: str = "gemini-2.5-flash"  # used by the gemini provider only
    generation_temperature: float = 0.0  # 0 => deterministic, grounding-friendly synthesis

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # Behavior.
    retrieval_top_k: int = 5
    confidence_abstain_threshold: float = 0.55

    # Retrieval fusion + gate (M2).
    rrf_k: int = 60  # Reciprocal Rank Fusion constant; larger => flatter rank weighting.
    # Relevance gate: drop a candidate unless dense cosine similarity clears this floor
    # OR the sparse (BM25) arm found an exact keyword hit. The floor is embedder-dependent
    # (hash vs openai live on different scales) — calibrated on the eval set in M5.
    retrieval_min_similarity: float = 0.15
    retrieval_candidate_pool: int = 20  # per-arm over-fetch before fusion; >= top_k

    # Ingestion.
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 150


@lru_cache
def get_settings() -> Settings:
    return Settings()
