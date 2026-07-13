"""Token usage + an *approximate* cost estimate for the eval table.

Scope note (deliberate): this is an eval-time **estimate** — measured token counts ×
the model's published list price — not the production cost path. Per-request cost
accounting wired into ``Answer.cost_usd`` across every provider is M7; this module
exists only so the M5 eval table can carry a real, defensible cost/1k figure instead
of a blank. It is marked approximate everywhere it surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass

# Gemini 2.5 Flash standard (paid-tier) list price, per 1M tokens, as published at
# https://ai.google.dev/gemini-api/docs/pricing (captured 2026-07-13). Output includes
# thinking tokens — we run thinking_budget=0, so output is pure answer/judge tokens.
GEMINI_2_5_FLASH_INPUT_USD_PER_1M = 0.30
GEMINI_2_5_FLASH_OUTPUT_USD_PER_1M = 2.50


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


def estimate_cost_usd(usage: TokenUsage) -> float:
    """List-price cost of one query's token usage (generation + judge), in USD."""
    return (
        usage.input_tokens / 1_000_000 * GEMINI_2_5_FLASH_INPUT_USD_PER_1M
        + usage.output_tokens / 1_000_000 * GEMINI_2_5_FLASH_OUTPUT_USD_PER_1M
    )
