"""Postgres + pgvector access: connection, schema, and idempotent upsert.

Ingestion is safe to re-run: each chunk has a deterministic id, so re-ingesting the
same docs UPDATEs rows instead of duplicating them. content_hash lets us tell whether
a chunk actually changed between runs (used later for freshness/re-embedding).
"""

from __future__ import annotations

from collections.abc import Iterable

import psycopg
from pgvector.psycopg import register_vector

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.knowledge.models import KnowledgeUnit


def get_conn(settings: Settings | None = None) -> psycopg.Connection:
    s = settings or get_settings()
    conn = psycopg.connect(s.database_url)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dim: int) -> None:
    """Create the extension, table, and vector index if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS knowledge_units (
                id                TEXT PRIMARY KEY,
                content           TEXT NOT NULL,
                source_uri        TEXT NOT NULL,
                section           TEXT,
                chunk_index       INTEGER NOT NULL DEFAULT 0,
                content_hash      TEXT NOT NULL,
                source_updated_at TIMESTAMPTZ,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
                embedding         vector({int(dim)})
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS knowledge_units_embedding_idx
            ON knowledge_units USING hnsw (embedding vector_cosine_ops);
            """
        )
    conn.commit()


def upsert_units(conn: psycopg.Connection, units: Iterable[KnowledgeUnit]) -> int:
    """Insert or update knowledge units by id. Returns the number written."""
    rows = list(units)
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO knowledge_units
                (id, content, source_uri, section, chunk_index, content_hash,
                 source_updated_at, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                content           = EXCLUDED.content,
                source_uri        = EXCLUDED.source_uri,
                section           = EXCLUDED.section,
                chunk_index       = EXCLUDED.chunk_index,
                content_hash      = EXCLUDED.content_hash,
                source_updated_at = EXCLUDED.source_updated_at,
                embedding         = EXCLUDED.embedding;
            """,
            [
                (
                    u.id,
                    u.content,
                    u.source_uri,
                    u.section,
                    u.chunk_index,
                    u.content_hash,
                    u.source_updated_at,
                    u.embedding,
                )
                for u in rows
            ],
        )
    conn.commit()
    return len(rows)
