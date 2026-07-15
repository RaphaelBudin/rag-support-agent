"""Faithfulness measurement without a human in the loop: LLM-as-judge entailment.

The question M5 has to answer is "are the answers *grounded* — is every statement
traceable to the retrieved context, or is the model leaking parametric memory?" — and
answer it as a number, reproducibly, with no human rater.

The design that makes this defensible:

  - **Claim-level entailment, not a holistic score.** The answer is broken into atomic
    factual claims and each is judged SUPPORTED / NOT_SUPPORTED against the context. A
    holistic "is this grounded 1-5" invites the judge to reward fluent, plausible prose;
    per-claim entailment forces it to point at evidence for each assertion.

  - **The judge never sees the question.** It sees only the CONTEXT passages and the
    ANSWER. This is the crux: faithfulness ≠ correctness. We are not asking "is this the
    right answer to the user" — we are asking "is every sentence supported by the supplied
    text." Withholding the question (and forbidding outside knowledge) stops the judge from
    rewarding a claim that is *true in the world* but *absent from the context* — which is
    exactly the parametric-memory leak grounding is supposed to prevent.

  - **Deterministic, house style.** temperature=0 and thinking off — the same knobs the
    generator uses. Judging entailment over supplied text is a narrow task, not a
    reasoning-heavy one.

This is the same primitive M4 left pluggable as ``generation.answer._grounding_factor``:
an LLM self-eval scoring entailment of the draft by its context. Here it is the eval-time
*measurement*; the identical call can later be wired as the online <1.0 grounding factor.

**What breaks / honest limit.** The judge is itself an LLM, and using Gemini to judge
Gemini risks *correlated error* — a model may rate its own phrasing as grounded. Mitigations:
the judge's task (entailment against given text) is far narrower and easier than open
generation, so it is more reliable than the thing it checks; temperature 0 + a strict rubric
cut variance. A cross-family judge (a non-Gemini model) removes the correlation outright, and
``OpenAIJudge`` provides one — so the cross-family pairing (generate with one family, judge with
the other) is now a config swap, gated only on having quota on both. LLM-gated by construction:
keyless, ``ExtractiveGenerator`` is faithful *by construction* (verbatim echo), so there is
nothing to measure — the harness reports that path as grounded-by-construction rather than
paying to judge a tautology.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.eval.cost import TokenUsage

_JUDGE_INSTRUCTION = (
    "You are a strict grounding checker. You are given CONTEXT passages and an ANSWER that "
    "was written from them. Check whether every factual claim in the ANSWER is supported by "
    "the CONTEXT.\n"
    "Rules:\n"
    "- Judge each claim ONLY against the CONTEXT below. Do NOT use outside knowledge and do "
    "NOT consider whether a claim is true in the real world — only whether the CONTEXT states "
    "or directly implies it.\n"
    "- Break the ANSWER into its distinct factual claims. Ignore citation markers like [1].\n"
    "- Ignore generic filler that makes no factual claim (e.g. 'Here is the answer:').\n"
    "- For each claim, output 'SUPPORTED' if the CONTEXT supports it, else 'NOT_SUPPORTED'.\n"
    'Return ONLY a JSON array, no prose: '
    '[{"claim": "<short paraphrase>", "verdict": "SUPPORTED"|"NOT_SUPPORTED"}]'
)


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool


@dataclass
class FaithfulnessResult:
    verdicts: list[ClaimVerdict]
    usage: TokenUsage

    @property
    def n_claims(self) -> int:
        return len(self.verdicts)

    @property
    def n_supported(self) -> int:
        return sum(1 for v in self.verdicts if v.supported)


class Judge(Protocol):
    def judge(self, answer_text: str, context: str) -> FaithfulnessResult: ...


def _parse_verdicts(raw: str) -> list[ClaimVerdict]:
    """Parse the judge's JSON array, tolerating ```json fences and surrounding prose."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    else:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            text = text[start : end + 1]
    data = json.loads(text)
    verdicts: list[ClaimVerdict] = []
    for item in data:
        verdict = str(item.get("verdict", "")).strip().upper()
        verdicts.append(
            ClaimVerdict(claim=str(item.get("claim", "")).strip(), supported=verdict == "SUPPORTED")
        )
    return verdicts


class GeminiJudge:
    """LLM-as-judge over Google Gemini, mirroring ``generation.generators.GeminiGenerator``."""

    def __init__(self, model: str, api_key: str | None, temperature: float = 0.0) -> None:
        from google import genai

        self.model = model
        self.temperature = temperature
        self._client = genai.Client(api_key=api_key)

    def judge(self, answer_text: str, context: str) -> FaithfulnessResult:
        from google.genai import types

        prompt = f"{_JUDGE_INSTRUCTION}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer_text}"
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        usage = TokenUsage(
            input_tokens=getattr(resp.usage_metadata, "prompt_token_count", 0) or 0,
            output_tokens=getattr(resp.usage_metadata, "candidates_token_count", 0) or 0,
        )
        return FaithfulnessResult(_parse_verdicts(resp.text or "[]"), usage)


class OpenAIJudge:
    """LLM-as-judge over OpenAI chat completions, mirroring ``generators.OpenAIGenerator``.

    Same entailment prompt, temperature, and JSON contract as ``GeminiJudge`` — the judge's
    task is fixed, only the model changes. Pairing an OpenAI judge with a Gemini generator
    (or vice-versa) is the *cross-family* check the write-up names: a judge from a different
    family cannot rate its own phrasing as grounded, removing the correlated-error risk.
    """

    def __init__(self, model: str, api_key: str | None, temperature: float = 0.0) -> None:
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key)

    def judge(self, answer_text: str, context: str) -> FaithfulnessResult:
        prompt = f"{_JUDGE_INSTRUCTION}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer_text}"
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = resp.usage
        token_usage = TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return FaithfulnessResult(_parse_verdicts(resp.choices[0].message.content or "[]"), token_usage)


def get_judge(settings: Settings | None = None) -> Judge | None:
    """An LLM judge when the config can support it, else ``None`` (keyless → skip).

    Faithfulness needs an LLM, so it is gated exactly like real synthesis — matched to the
    active ``llm_provider``. Returning ``None`` (rather than raising) lets the harness compute
    every keyless metric and simply report faithfulness as skipped.
    """
    s = settings or get_settings()
    if s.llm_provider == "gemini" and s.gemini_api_key:
        model = s.generation_model if s.generation_model.startswith("gemini") else "gemini-2.5-flash"
        return GeminiJudge(model, s.gemini_api_key)
    if s.llm_provider == "openai" and s.openai_api_key:
        model = s.generation_model if s.generation_model.startswith("gpt") else "gpt-4o-mini"
        return OpenAIJudge(model, s.openai_api_key)
    return None
