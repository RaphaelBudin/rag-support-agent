"""Unit tests for the heading-aware chunker (pure stdlib — no DB, no API)."""

from rag_support_agent.ingestion.chunker import chunk_markdown

DOC = """# API keys

Intro paragraph about keys.

## Rotating a key

To rotate, click Rotate. The old secret stays valid for 24 hours.

## Revoking a key

Click Revoke to disable a key at once.
"""


def test_sections_become_separate_chunks():
    chunks = chunk_markdown(DOC, target=1200, overlap=100)
    sections = {c.section for c in chunks}
    assert "API keys > Rotating a key" in sections
    assert "API keys > Revoking a key" in sections


def test_indices_are_sequential():
    chunks = chunk_markdown(DOC)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_long_section_is_windowed_with_size_bound():
    body = "\n\n".join(f"Paragraph number {i} with some filler text." for i in range(60))
    doc = f"# Big\n\n{body}"
    chunks = chunk_markdown(doc, target=400, overlap=80)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 + 200 for c in chunks)  # target + generous slack
    assert all(c.section == "Big" for c in chunks)


def test_empty_input_yields_no_chunks():
    assert chunk_markdown("") == []
