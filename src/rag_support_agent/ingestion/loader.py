"""Load raw documents from a directory into a normalized form.

M1 supports Markdown; the loader returns the text plus provenance metadata,
including the source file's modification time — that timestamp is what later
powers freshness/decay (M6), so we capture it at ingest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LoadedDoc:
    source_uri: str
    text: str
    source_updated_at: datetime


def load_dir(source_dir: str | Path, patterns: tuple[str, ...] = ("*.md",)) -> list[LoadedDoc]:
    """Load every matching document under ``source_dir`` (recursively)."""
    root = Path(source_dir)
    if not root.exists():
        raise FileNotFoundError(f"source dir not found: {root}")

    docs: list[LoadedDoc] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            docs.append(
                LoadedDoc(
                    source_uri=str(path.relative_to(root)),
                    text=text,
                    source_updated_at=mtime,
                )
            )
    return docs
