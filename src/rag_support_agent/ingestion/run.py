"""Ingestion entrypoint:  python -m rag_support_agent.ingestion.run --source data/sample_docs

Loads documents, chunks them (heading-aware, explicit size+overlap), builds
KnowledgeUnits with freshness metadata, embeds, and upserts into pgvector.

TODO(M1): implement loader + chunker + embed + upsert.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the knowledge base.")
    parser.add_argument("--source", required=True, help="Path to a docs directory.")
    args = parser.parse_args()
    print(f"[ingest] TODO: load, chunk, embed, upsert from {args.source!r}")


if __name__ == "__main__":
    main()
