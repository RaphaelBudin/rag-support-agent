"""Answer generators behind a tiny interface (mirrors ``retrieval.embeddings``).

Two implementations, selected by ``LLM_PROVIDER``:

  - ExtractiveGenerator: the keyless default. It does not *synthesize* — it returns the
    top retrieved passages verbatim, each tagged with an inline ``[n]`` citation marker.
    Grounding is guaranteed *by construction* (it can only echo retrieved text, so it
    cannot hallucinate), which is exactly why it makes a safe zero-key default: a reviewer
    can run the whole pipeline (retrieve -> ground -> cite -> abstain) with no API key.

  - GeminiGenerator: the real synthesizer. Same grounding contract, enforced by a strict
    prompt (only the provided context, cite every claim, refuse when the context does not
    answer) at temperature 0. This is prompt-enforced grounding, not proven grounding —
    which is why M5 measures answer *faithfulness* as a number.

Both paths share the *same* passage numbering (``format_context``) and the *same* citation
extractor (``parse_citations``), so a ``[n]`` marker means the same thing regardless of who
produced the text. That single shared convention is what lets ``answer.build_answer`` turn
any generator's output into a citation list without knowing which generator ran.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.knowledge.models import RetrievalResult

# The generator emits this exact token when the retrieved context does not answer the
# question. It is how the model *refuses parametric memory* — the M3 grounding guarantee,
# distinct from the calibrated, confidence-based abstention that M4 adds on top.
SENTINEL = "INSUFFICIENT_CONTEXT"

# User-facing text for the two ways an answer can be withheld.
INSUFFICIENT_CONTEXT_MESSAGE = (
    "The retrieved sources don't contain enough information to answer that confidently."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class GenResult:
    """What a generator returns: the answer text and whether it refused.

    ``abstained=True`` means the generator itself decided the context is insufficient
    (the Gemini sentinel). The empty-retrieval abstention is handled one level up, in
    ``answer.build_answer``, before any generator is even called.
    """

    text: str
    abstained: bool = False


def format_context(results: list[RetrievalResult]) -> str:
    """Render retrieved passages as a numbered block: ``[1] (source :: section)\\n<body>``.

    The number is the citation handle. Shared by both generators so the ``[n]`` a model
    writes lines up with the passage a human (and ``parse_citations``) reads.
    """
    blocks = []
    for i, r in enumerate(results, start=1):
        label = r.unit.source_uri
        if r.unit.section:
            label = f"{label} :: {r.unit.section}"
        blocks.append(f"[{i}] ({label})\n{r.unit.content.strip()}")
    return "\n\n".join(blocks)


def parse_citations(text: str, n: int) -> list[int]:
    """Extract the ``[k]`` markers actually present in ``text``, as 1-based indices.

    Unique, in first-appearance order, and filtered to the valid range ``1..n`` — a model
    that hallucinates a ``[9]`` when only 5 passages were supplied gets that stray marker
    dropped rather than turned into a dangling citation.
    """
    seen: list[int] = []
    for m in _CITATION_RE.finditer(text):
        k = int(m.group(1))
        if 1 <= k <= n and k not in seen:
            seen.append(k)
    return seen


class Generator(Protocol):
    def generate(self, query: str, results: list[RetrievalResult]) -> GenResult: ...


class ExtractiveGenerator:
    """Keyless default: echo the top retrieved passages verbatim, each ``[n]``-tagged.

    Not a synthesizer. It proves the end-to-end contract (grounded + cited + abstains) with
    zero API dependency and zero hallucination risk, since every character of the answer is
    copied from a retrieved unit. The real synthesis lives in ``GeminiGenerator``.
    """

    # Cap how many passages we echo so the answer stays readable; the rest still retrieved,
    # just not shown. The gated top result is the primary answer; the next few support it.
    _MAX_PASSAGES = 3

    def generate(self, query: str, results: list[RetrievalResult]) -> GenResult:
        top = results[: self._MAX_PASSAGES]
        lines = ["Answering from the retrieved documentation (verbatim excerpts):", ""]
        for i, r in enumerate(top, start=1):
            lines.append(f"[{i}] {r.unit.content.strip()}")
            lines.append("")
        lines.append("Sources:")
        for i, r in enumerate(top, start=1):
            label = r.unit.source_uri
            if r.unit.section:
                label = f"{label} :: {r.unit.section}"
            lines.append(f"  [{i}] {label}")
        return GenResult(text="\n".join(lines).strip())


_GEMINI_INSTRUCTION = (
    "You are a support assistant. Answer the user's question using ONLY the numbered "
    "context passages provided below. Do not use any outside or prior knowledge.\n"
    "- Ground every statement in the context and cite it with the passage number in "
    "square brackets, e.g. [1] or [2].\n"
    f"- If the context does not contain enough information to answer, reply with exactly: {SENTINEL}\n"
    "- Be concise and specific; do not add information that is not in the context."
)


class GeminiGenerator:
    """Real synthesis via Google Gemini (``google-genai``), grounded by prompt.

    temperature=0 and thinking disabled: grounded synthesis over supplied passages is not a
    reasoning-heavy task, so we drop the thinking budget for a deterministic, lower-latency,
    lower-cost answer. Grounding here is *prompt-enforced* (only the context, cite, or emit
    the sentinel) — an LLM can still drift, which is why faithfulness becomes a measured
    number in M5 rather than an assumed property.
    """

    def __init__(self, model: str, api_key: str | None, temperature: float) -> None:
        from google import genai

        self.model = model
        self.temperature = temperature
        self._client = genai.Client(api_key=api_key)

    def generate(self, query: str, results: list[RetrievalResult]) -> GenResult:
        from google.genai import types

        prompt = (
            f"{_GEMINI_INSTRUCTION}\n\nContext:\n{format_context(results)}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (resp.text or "").strip()
        # Empty output or the refusal sentinel => the model declined to answer from context.
        if not text or text.startswith(SENTINEL):
            return GenResult(text=INSUFFICIENT_CONTEXT_MESSAGE, abstained=True)
        return GenResult(text=text)


def get_generator(settings: Settings | None = None) -> Generator:
    s = settings or get_settings()
    if s.llm_provider == "extractive":
        return ExtractiveGenerator()
    if s.llm_provider == "gemini":
        # Fall back to a Gemini model name if the config still holds a non-Gemini default.
        model = s.generation_model if s.generation_model.startswith("gemini") else "gemini-2.5-flash"
        return GeminiGenerator(model, s.gemini_api_key, s.generation_temperature)
    raise ValueError(
        f"unsupported llm_provider: {s.llm_provider!r} (supported: 'extractive', 'gemini')"
    )
