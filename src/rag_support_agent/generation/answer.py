"""Answer generation with grounding, confidence scoring, and abstention.

The agent answers ONLY from retrieved context (no parametric memory), attaches
citations, and computes a confidence signal. Below the threshold it abstains
instead of guessing — one confident hallucination costs more trust than ten
honest "I don't know"s.

TODO(M3/M4): implement synthesis + confidence calibration.
"""

from __future__ import annotations

from rag_support_agent.knowledge.models import Answer, KnowledgeUnit


def compute_confidence(query: str, retrieved: list[tuple[KnowledgeUnit, float]]) -> float:
    """Confidence in [0,1] from retrieval score spread + grounding check.

    Signals to combine (M4):
      - top score and gap to the next result (a clear winner => higher confidence)
      - whether the drafted answer is entailed by the retrieved context
      - optional lightweight self-eval
    """
    raise NotImplementedError


def answer_question(query: str, abstain_threshold: float = 0.55, top_k: int = 5) -> Answer:
    """End-to-end: retrieve -> ground -> cite -> score -> answer or abstain.

    Also attaches stale-source flags (knowledge.freshness) and cost/latency
    (observability) so the caller gets the full trust picture in one object.
    """
    raise NotImplementedError
