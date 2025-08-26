"""LLM-based AspectCritic example for Ragas.

What this measures
    Binary pass/fail judgment against a natural-language criterion (aspect),
    e.g., tone, safety, conciseness.

When to use
    - Quick compliance checks (professional tone, prohibited content).
    - CI gates for stylistic/branding constraints.

Trade-offs
    - Binary: low resolution but simple to operationalize.
    - Requires thoughtful definition text; LLM cost applies.

Implementation notes
    - Name and definition form the evaluator; prompts can be adapted
      for domain-specific expectations.

Run
    uv run docs/ragas/examples/aspect_critic.py
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic


def build_sample() -> SingleTurnSample:
    return SingleTurnSample(
        user_input="Provide a brief professional email greeting",
        response="Hey buddy, what's up?",
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    metric = AspectCritic(
        name="professional_tone",
        llm=llm,
        definition=(
            "Return 1 if the response uses a professional tone appropriate for a"
            " business email greeting; otherwise return 0."
        ),
    )
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("AspectCritic (professional tone):", score)


if __name__ == "__main__":
    main()
