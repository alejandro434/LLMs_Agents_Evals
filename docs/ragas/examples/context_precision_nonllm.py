"""Non-LLM Context Precision with Reference example for Ragas.

What this measures
    Precision of retrieved contexts against a known set of reference
    contexts (signal-to-noise ratio). Non-LLM, deterministic.

When to use
    - RAG retrieval validation when you have labeled reference contexts.
    - CI/CD guardrail for retriever regressions.

Trade-offs
    - Requires `reference_contexts` labels; does not judge generation.
    - Does not capture semantic relevance beyond string similarity.

Implementation notes
    - Uses `NonLLMStringSimilarity` under the hood.
    - Pairs well with recall to assess completeness.

Run
    uv run docs/ragas/examples/context_precision_nonllm.py
"""

# %%
from __future__ import annotations

from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference


def build_sample() -> SingleTurnSample:
    """Create a sample including reference contexts for non-LLM precision."""
    return SingleTurnSample(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris, France.",
        retrieved_contexts=[
            "The Eiffel Tower is in Paris, France.",  # relevant
            "The Great Wall is in China.",  # noise
            "Louvre Museum is also in Paris.",  # neutral-ish
        ],
        reference_contexts=[
            "The Eiffel Tower is in Paris, France.",
            "It is a landmark in Paris.",
        ],
        reference="The Eiffel Tower is in Paris, France.",
    )


def main() -> None:
    metric = NonLLMContextPrecisionWithReference()
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("NonLLMContextPrecisionWithReference:", score)


if __name__ == "__main__":
    main()
