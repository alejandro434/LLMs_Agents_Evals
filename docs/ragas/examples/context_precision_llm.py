"""LLM-based Context Precision without reference example for Ragas.

What this measures
    Proportion of retrieved context chunks that are relevant to the
    produced `response` (signal vs. noise) without requiring gold labels.

When to use
    - Early-stage RAG experiments without annotated contexts.
    - Monitoring retrieval noise in production.

Trade-offs
    - Uses an LLM judge: cost, latency, and permissions apply.
    - Relevance is judged against the model's response (may reflect its
      errors). Use "with reference" variants when gold answers exist.

Implementation notes
    - Provide a capable judge LLM; adjust prompts via `get_prompts()` if
      domain adaptation is needed.

Run
    uv run docs/ragas/examples/context_precision_llm.py

Requires
    OPENAI_API_KEY set. Uses LangChain wrappers.
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Where is Mount Everest located?",
        response="Mount Everest is located on the border between Nepal and China.",
        retrieved_contexts=[
            "Mount Everest lies in the Himalayas on the Nepal–China border.",
            "K2 is located between Pakistan and China.",
            "The Himalayas span five countries: Bhutan, India, Nepal, China, and Pakistan.",
        ],
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = LLMContextPrecisionWithoutReference(llm=llm)
    sample = build_sample()
    try:
        score = metric.single_turn_score(sample)
        print("LLMContextPrecisionWithoutReference:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
