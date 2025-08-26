"""Non-LLM Context Recall example for Ragas.

What this measures
    Recall of retrieved contexts against labeled `reference_contexts`.
    Measures completeness of retrieval (did we fetch all needed docs?).

When to use
    - RAG experiments with curated gold contexts.
    - Detect recall regressions when changing retriever parameters.

Trade-offs
    - Requires labeled reference contexts; no semantic disambiguation.
    - Pair with precision to balance completeness vs. noise.

Implementation notes
    - Deterministic; uses classical similarity.
    - Great for offline retriever benchmarking.

Run
    uv run docs/ragas/examples/context_recall_nonllm.py
"""

# %%
from __future__ import annotations

from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextRecall


def build_sample() -> SingleTurnSample:
    """Create a sample including reference contexts for non-LLM recall."""
    return SingleTurnSample(
        user_input="Who painted the Mona Lisa?",
        response="Leonardo da Vinci painted the Mona Lisa.",
        retrieved_contexts=[
            "The Mona Lisa is a portrait by Leonardo da Vinci.",
            "The painting is housed in the Louvre Museum.",
        ],
        reference_contexts=[
            "The Mona Lisa was painted by Leonardo da Vinci.",
            "It is a Renaissance masterpiece.",
        ],
        reference="The Mona Lisa was painted by Leonardo da Vinci.",
    )


def main() -> None:
    metric = NonLLMContextRecall()
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("NonLLMContextRecall:", score)


if __name__ == "__main__":
    main()
