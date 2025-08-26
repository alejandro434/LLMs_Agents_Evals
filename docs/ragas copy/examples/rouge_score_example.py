"""Deterministic RougeScore example for Ragas.

What this measures
    ROUGE measures n-gram recall/precision/F1 between `response` and
    `reference`. Often used in summarization evaluation.

When to use
    - Summarization pipelines where lexical overlap is meaningful.
    - Low-cost, reproducible CI checks.

Trade-offs
    - Penalizes paraphrases and stylistic variation.
    - For semantic equivalence, consider `SemanticSimilarity`.

Implementation notes
    - Requires the `rouge-score` dependency.
    - No LLMs required.

Run
    uv run docs/ragas/examples/rouge_score_example.py
"""

# %%
from __future__ import annotations

from ragas import SingleTurnSample
from ragas.metrics import RougeScore


def build_sample() -> SingleTurnSample:
    """Create a simple single-turn sample for ROUGE.

    ROUGE measures n-gram recall/precision/F1 between response and reference.
    """
    return SingleTurnSample(
        user_input="Summarize the Mona Lisa",
        response="The Mona Lisa is a Renaissance portrait by Leonardo da Vinci.",
        reference=(
            "Leonardo da Vinci painted the Mona Lisa, a famous Renaissance portrait."
        ),
        retrieved_contexts=[],
    )


def main() -> None:
    metric = RougeScore()
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("RougeScore:", score)


if __name__ == "__main__":
    main()
