"""Deterministic ExactMatch example for Ragas.

What this measures
    ExactMatch returns 1.0 when `response` is a character-for-character
    match with `reference`, otherwise 0.0. It is fully deterministic and
    extremely fast.

When to use
    - Structured outputs: router decisions, tool arguments, IDs, keywords.
    - CI/CD checks where you need stability and no LLM costs.
    - AWARE: validating router direct responses or strict contract fields.

Trade-offs
    - Ignores paraphrasing and semantics. "Paris" vs. "City of Paris"
      will score 0. Prefer semantic metrics for flexible wording.

Implementation notes
    - Does a direct string equality check on `response` and `reference`.
    - No LLMs or embeddings required (cheap and reproducible).

Run
    uv run docs/ragas/examples/exact_match.py
"""

# %%
from __future__ import annotations

from ragas import SingleTurnSample
from ragas.metrics import ExactMatch


def build_sample() -> SingleTurnSample:
    """Create a simple single-turn sample.

    ExactMatch requires `response` and `reference`.
    """
    return SingleTurnSample(
        user_input="What is the capital of France?",
        response="Paris",
        reference="Paris",
        retrieved_contexts=[],
    )


def main() -> None:
    metric = ExactMatch()
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("ExactMatch score:", score)


if __name__ == "__main__":
    main()
