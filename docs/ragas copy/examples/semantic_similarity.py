"""SemanticSimilarity example for Ragas (requires embeddings).

What this measures
    Embedding-based cosine similarity between `response` and `reference`.
    Captures paraphrasing and meaning alignment beyond surface forms.

When to use
    - Reasoner and planner outputs where wording varies but meaning should
      match ground truth.
    - AWARE: system-level checks for semantic fidelity of final outputs.

Trade-offs
    - Requires embeddings (LLM cost and permissions). Results depend on the
      embedding model quality and domain alignment.
    - Less explainable than claim-level metrics (e.g., `FactualCorrectness`).

Implementation notes
    - Wraps a LangChain embedding model using `LangchainEmbeddingsWrapper`.
    - Consider normalizing inputs and ensuring consistent casing/tokenization.

Run
    uv run docs/ragas/examples/semantic_similarity.py
"""

# %%
from __future__ import annotations

import os

from langchain_openai import OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import SemanticSimilarity


def build_sample() -> SingleTurnSample:
    """Create a simple single-turn sample for semantic similarity."""
    return SingleTurnSample(
        user_input="Explain photosynthesis in brief",
        response=(
            "Photosynthesis converts light energy into chemical energy in plants,"
            " producing glucose and oxygen."
        ),
        reference=(
            "Plants use light to make chemical energy, generating sugars and oxygen"
            " through photosynthesis."
        ),
        retrieved_contexts=[],
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    metric = SemanticSimilarity(embeddings=embeddings)
    sample = build_sample()
    try:
        score = metric.single_turn_score(sample)
        print("SemanticSimilarity:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
