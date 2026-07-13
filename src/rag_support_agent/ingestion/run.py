"""Ingestion entrypoint.

    python -m rag_support_agent.ingestion.run --source data/sample_docs
    python -m rag_support_agent.ingestion.run --source data/sample_docs --dry-run

Pipeline: load docs -> heading-aware chunk -> build KnowledgeUnits (deterministic id
+ content hash + freshness timestamp) -> embed -> upsert into pgvector. Idempotent.
"""

from __future__ import annotations

import argparse
import hashlib

from rag_support_agent.config import get_settings
from rag_support_agent.ingestion.chunker import chunk_markdown
from rag_support_agent.ingestion.loader import load_dir
from rag_support_agent.knowledge.models import KnowledgeUnit
from rag_support_agent.retrieval.embeddings import get_embedder


def _unit_id(source_uri: str, chunk_index: int) -> str:
    return hashlib.sha1(f"{source_uri}#{chunk_index}".encode()).hexdigest()[:16]


def build_units(source_dir: str, target: int, overlap: int) -> list[KnowledgeUnit]:
    """Load and chunk a directory into KnowledgeUnits (without embeddings)."""
    units: list[KnowledgeUnit] = []
    for doc in load_dir(source_dir):
        for chunk in chunk_markdown(doc.text, target=target, overlap=overlap):
            units.append(
                KnowledgeUnit(
                    id=_unit_id(doc.source_uri, chunk.index),
                    content=chunk.text,
                    source_uri=doc.source_uri,
                    section=chunk.section or None,
                    chunk_index=chunk.index,
                    content_hash=hashlib.sha256(chunk.text.encode()).hexdigest(),
                    source_updated_at=doc.source_updated_at,
                )
            )
    return units


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the knowledge base.")
    parser.add_argument("--source", default="data/sample_docs", help="Docs directory.")
    parser.add_argument("--dry-run", action="store_true", help="Chunk + embed but don't touch the DB.")
    args = parser.parse_args()

    settings = get_settings()
    units = build_units(args.source, settings.chunk_target_chars, settings.chunk_overlap_chars)
    n_docs = len({u.source_uri for u in units})
    print(f"[ingest] {n_docs} docs -> {len(units)} chunks")

    # Embed the heading path *with* the body. The heading ("... > Rotating a key") is
    # the strongest topic label a chunk has, and a chunk whose error code lives only in
    # its heading would otherwise be near-invisible to dense search. The stored/cited
    # content stays the pure body — only the embedding input is augmented.
    embedder = get_embedder(settings)
    embed_inputs = [f"{u.section}\n\n{u.content}" if u.section else u.content for u in units]
    vectors = embedder.embed(embed_inputs)
    for unit, vec in zip(units, vectors):
        unit.embedding = vec
    print(f"[ingest] embedded with provider={settings.embedding_provider} dim={embedder.dim}")

    if args.dry_run:
        print("[ingest] --dry-run: skipping DB write")
        for u in units[:3]:
            print(f"  - {u.source_uri} :: {u.section} :: {len(u.content)} chars")
        return

    from rag_support_agent.knowledge.db import get_conn, init_schema, upsert_units

    conn = get_conn(settings)
    init_schema(conn, settings.embedding_dim)
    written = upsert_units(conn, units)
    conn.close()
    print(f"[ingest] upserted {written} knowledge units into pgvector")


if __name__ == "__main__":
    main()
