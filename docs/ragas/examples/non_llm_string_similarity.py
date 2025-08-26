"""Deterministic NonLLMStringSimilarity example for Ragas.

What this measures
    Classical string distance similarities (e.g., Levenshtein/Jaro)
    between `response` and `reference`.

When to use
    - Quick lexical similarity proxy without LLM cost.
    - Detect minor edits/typos or near-duplicate responses.

Trade-offs
    - No semantic awareness; paraphrases may look dissimilar.
    - For meaning-preserving variations, use `SemanticSimilarity`.

Implementation notes
    - Requires `rapidfuzz`.
    - Deterministic; good for CI and baselines.

Run
    uv run docs/ragas/examples/non_llm_string_similarity.py
"""

# %%
from __future__ import annotations

from ragas import SingleTurnSample
from ragas.metrics import NonLLMStringSimilarity


def build_sample() -> SingleTurnSample:
    """Create sample for non-LLM string similarity.

    Measures classical string similarity between response and reference.
    """
    return SingleTurnSample(
        user_input="Define machine learning",
        response=(
            "Machine learning is a field of study that gives computers the ability"
            " to learn without being explicitly programmed."
        ),
        reference=(
            "Machine learning is a field that enables computers to learn without"
            " explicit programming."
        ),
        retrieved_contexts=[],
    )


def main() -> None:
    metric = NonLLMStringSimilarity()
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("NonLLMStringSimilarity:", score)


if __name__ == "__main__":
    main()
