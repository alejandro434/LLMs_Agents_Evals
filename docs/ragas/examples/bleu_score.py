"""Deterministic BleuScore example for Ragas.

What this measures
    BLEU computes n-gram precision overlap between `response` and
    `reference`. It is widely used in MT/summarization and is fully
    deterministic.

When to use
    - Regression testing for summarization-style outputs with similar
      phrasing.
    - Quick proxy for overlap without LLM costs.

Trade-offs
    - Sensitive to wording changes and synonyms; low correlation with
      human judgments when paraphrasing is common.
    - Prefer semantic metrics for flexible language.

Implementation notes
    - Uses n-gram match precision. No LLMs required.

Run
    uv run docs/ragas/examples/bleu_score.py
"""

# %%
from __future__ import annotations

from ragas import SingleTurnSample
from ragas.metrics import BleuScore


def build_sample() -> SingleTurnSample:
    """Create a simple single-turn sample for BLEU.

    BLEU compares n-gram overlap between response and reference.
    """
    return SingleTurnSample(
        user_input="Describe the Eiffel Tower",
        response="The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        reference=(
            "The Eiffel Tower is an iron lattice tower located in Paris, France."
        ),
        retrieved_contexts=[],
    )


def main() -> None:
    metric = BleuScore()
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("BleuScore:", score)


if __name__ == "__main__":
    main()
