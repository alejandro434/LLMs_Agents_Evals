"""LLM-based NoiseSensitivity example for Ragas.

What this measures
    Robustness to irrelevant context: how often the model produces incorrect
    claims when noise is present in the retrieved contexts.

When to use
    - Stress-testing RAG systems for distraction resilience.
    - Measuring susceptibility to misleading evidence.

Trade-offs
    - LLM cost/latency; analysis focuses on claim-level deltas.

Implementation notes
    - Decomposes ground-truth into statements, checks model response under
      noisy contexts, and estimates sensitivity.

Run
    uv run docs/ragas/examples/noise_sensitivity.py

Requires
    OPENAI_API_KEY set. Uses LangChain wrappers.
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import NoiseSensitivity


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Explain the role of a retriever in a RAG pipeline",
        response=(
            "A retriever finds relevant documents which the generator then uses"
            " to ground its answer."
        ),
        reference=(
            "The retriever fetches relevant documents based on the query for the"
            " generator to use when forming an answer."
        ),
        retrieved_contexts=[
            "RAG leverages a retriever to find relevant context.",  # relevant
            "The Amazon rainforest is the largest tropical rainforest.",  # noise
        ],
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = NoiseSensitivity(llm=llm)
    sample = build_sample()
    try:
        score = metric.single_turn_score(sample)
        print("NoiseSensitivity:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
