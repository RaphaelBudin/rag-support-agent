"""Heading-aware Markdown chunking.

Why not fixed-size chunks: support docs are structured, and a heading ("Rotating a
key") is the single strongest signal of what a passage is *about*. So we split on
headings first — each chunk stays inside one section and remembers its heading path,
which both improves retrieval and gives every answer a precise citation. Only when a
single section is longer than the target size do we window it (with overlap, on
paragraph boundaries) so no chunk is unmanageably large.

Pure-stdlib on purpose: this is the part most worth unit-testing, so it has zero
heavy dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    section: str  # heading path, e.g. "API keys > Rotating a key"
    text: str
    index: int


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split into (heading_path, body) pairs following the heading hierarchy."""
    sections: list[tuple[str, list[str]]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)

    for line in markdown.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            path = " > ".join(t for _, t in heading_stack)
            sections.append((path, []))
        else:
            if not sections:
                sections.append(("", []))  # preamble before any heading
            sections[-1][1].append(line)

    return [(path, "\n".join(lines).strip()) for path, lines in sections]


def _window_paragraphs(body: str, target: int, overlap: int) -> list[str]:
    """Pack paragraphs into <=target-char windows, carrying an overlap tail."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail  # start at a word
            current = f"{tail}\n\n{para}" if tail else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_markdown(markdown: str, target: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Chunk a Markdown document into heading-scoped, size-bounded chunks."""
    out: list[Chunk] = []
    for path, body in _split_sections(markdown):
        if not body:
            continue
        pieces = [body] if len(body) <= target else _window_paragraphs(body, target, overlap)
        for piece in pieces:
            out.append(Chunk(section=path, text=piece, index=len(out)))
    return out
