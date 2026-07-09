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
    # repo runs end-to-end with no API key; "openai" is the real one.
    embedding_provider: str = "hash"  # "hash" | "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536  # must match the provider's output and the DB column

    # Generation (used from M3 on).
    llm_provider: str = "openai"  # "openai" | "anthropic"
    generation_model: str = "gpt-4o-mini"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Behavior.
    retrieval_top_k: int = 5
    confidence_abstain_threshold: float = 0.55

    # Ingestion.
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 150


@lru_cache
def get_settings() -> Settings:
    return Settings()
