"""NVIDIA AnswerAccuracy example for Ragas.

What this measures
    Agreement between `response` and `reference` using two independent
    LLM judges, normalized to [0, 1]. Optimized for robustness.

When to use
    - Production dashboards requiring stable, low-variance scoring.
    - High-level accuracy KPIs when detailed reasoning traces are not needed.

Trade-offs
    - Less explainable than claim-level metrics (e.g., FactualCorrectness).
    - Lower token usage vs. multi-step judge prompts.

Implementation notes
    - Dual-judge architecture reduces variance from single-judge anomalies.

Run
    uv run docs/ragas/examples/answer_accuracy.py
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerAccuracy


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="What is the boiling point of water at sea level?",
        response="The boiling point of water at sea level is 100 degrees Celsius.",
        reference="Water boils at 100°C at standard atmospheric pressure.",
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = AnswerAccuracy(llm=llm)
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("AnswerAccuracy:", score)


if __name__ == "__main__":
    main()
