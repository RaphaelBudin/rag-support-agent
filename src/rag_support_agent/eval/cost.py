"""Token usage + an *approximate* cost estimate for the eval table.

Scope note (deliberate): this is an eval-time **estimate** — measured token counts ×
the model's published list price — not the production cost path. Per-request cost
accounting wired into ``Answer.cost_usd`` across every provider is M7; this module
exists only so the M5 eval table can carry a real, defensible cost/1k figure instead
of a blank. It is marked approximate everywhere it surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

# Published (paid-tier) list prices, USD per 1M tokens, as (input, output). Sources:
# Gemini 2.5 Flash — https://ai.google.dev/gemini-api/docs/pricing (captured 2026-07-13);
# gpt-4o-mini    — https://openai.com/api/pricing (captured 2026-07-15). Output excludes
# thinking tokens for both (we run thinking_budget=0 / a non-reasoning model), so it is pure
# answer/judge output. Matched by model-name prefix so the cost row is right for whichever
# provider actually ran, not silently the default's price on another model's tokens.
_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gpt-4o-mini": (0.15, 0.60),
}
_DEFAULT_MODEL = "gemini-2.5-flash"


@dataclass
class TokenUsage:
    """Prompt/response token counts, summable across the calls a query makes."""

    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


def _price_for(model: str | None) -> tuple[float, float]:
    """(input, output) $/1M for ``model``, matched by prefix; falls back to the default model."""
    name = model or _DEFAULT_MODEL
    for prefix, price in _PRICING_USD_PER_1M.items():
        if name.startswith(prefix):
            return price
    return _PRICING_USD_PER_1M[_DEFAULT_MODEL]


def estimate_cost_usd(usage: TokenUsage, model: str | None = None) -> float:
    """List-price cost of one query's token usage (generation or judge), in USD.

    ``model`` selects the price band (default: the Gemini generator's). Pass the model that
    actually produced ``usage`` so an OpenAI run isn't costed at Gemini's rates.
    """
    in_price, out_price = _price_for(model)
    return usage.input_tokens / 1_000_000 * in_price + usage.output_tokens / 1_000_000 * out_price
