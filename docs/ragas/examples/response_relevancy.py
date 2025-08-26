"""LLM-based ResponseRelevancy example for Ragas.

What this measures
    How pertinent the `response` is to the `user_input`. Uses an LLM to
    generate likely questions from the response and compares them via
    embeddings to the original query.

When to use
    - Ensure answers stay focused on the user question.
    - Catch verbose but off-topic responses.

Trade-offs
    - Requires both LLM and embeddings; cost and permissions needed.
    - Heuristic proxy (reverse QA). For deeper semantic alignment, also
      consider `SemanticSimilarity` or end-to-end metrics.

Implementation notes
    - Tunable strictness; domain-adapt prompts for better reliability.

Run
    uv run docs/ragas/examples/response_relevancy.py

Requires
    OPENAI_API_KEY set. Uses LangChain wrappers for LLM and embeddings.
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ResponseRelevancy


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="What is retrieval-augmented generation (RAG)?",
        response=(
            "RAG retrieves relevant documents and uses them to ground the answer"
            " produced by a generator."
        ),
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    metric = ResponseRelevancy(llm=llm, embeddings=embeddings)
    sample = build_sample()
    try:
        score = metric.single_turn_score(sample)
        print("ResponseRelevancy:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
