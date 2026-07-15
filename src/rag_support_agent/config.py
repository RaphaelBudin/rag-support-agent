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
    # runs end-to-end for a stranger; "gemini" and "openai" are the real synthesizers.
    llm_provider: str = "extractive"  # "extractive" (keyless) | "gemini" | "openai"
    generation_model: str = "gemini-2.5-flash"  # gemini path; openai path defaults to gpt-4o-mini
    generation_temperature: float = 0.0  # 0 => deterministic, grounding-friendly synthesis

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

    # Behavior.
    retrieval_top_k: int = 5
    # M4 abstention: below this confidence the agent abstains instead of answering.
    # Confidence is a retrieval *spread* (top-hit gap over the field) in ~[0, 0.3], NOT a
    # cosine floor — so the threshold lives on that spread scale. 0.12 is the measured
    # CLEAR/ambiguous boundary under a semantic embedder; provisional — M5 calibrates it
    # against the labeled eval set (abstention precision/recall).
    confidence_abstain_threshold: float = 0.12

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

    # Freshness / decay (M6). Age becomes a decay-*risk* score, NOT a staleness measurement
    # (no ground truth says a doc is wrong now). Two complementary signals, mirroring M4's
    # "relative transfers, absolute floor doesn't":
    #  - Absolute: freshness = 0.5 ** (age_days / half_life). A source is flagged "possibly
    #    stale" when it falls past ``freshness_stale_score`` (default 0.5 = one half-life old).
    #    ``half_life`` is a policy knob set to the domain's rate of change, not learned.
    #  - Relative: among the retrieved sources, one whose age is >= factor x the median peer
    #    age AND at least ``min_gap_days`` older is an age-outlier — transfers across absolute
    #    bands and stays silent on a uniform corpus (e.g. a fresh clone resets every mtime).
    freshness_half_life_days: float = 180.0
    freshness_stale_score: float = 0.5
    freshness_relative_factor: float = 2.0
    freshness_relative_min_gap_days: float = 90.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
