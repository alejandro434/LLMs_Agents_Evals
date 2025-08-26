"""TopicAdherenceScore example for Ragas.

What this measures
    Whether conversation topics remain within an allowed set (`reference_topics`).
    Produces precision/recall/F1 for in-scope topic handling.

When to use
    - Brand safety and compliance (e.g., avoid investment advice).
    - Domain-scoped assistants that must refuse out-of-scope queries.

Trade-offs
    - Requires an LLM for topic extraction/classification; cost applies.
    - Refusals can be configured and accounted for explicitly.

Implementation notes
    - Multi-turn metric using `MultiTurnSample`.

Run
    uv run docs/ragas/examples/topic_adherence.py
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics import TopicAdherenceScore


def build_sample() -> MultiTurnSample:
    messages = [
        HumanMessage(content="I need help with my bank account balance."),
        AIMessage(content="Sure, I can help with account inquiries."),
        HumanMessage(content="Also, what are good stocks to buy this week?"),
        AIMessage(
            content="I cannot provide investment advice. I can help with account info."
        ),
    ]
    return MultiTurnSample(
        user_input=messages,
        reference_topics=["bank account", "balance", "transactions", "support"],
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini", temperature=0))
    metric = TopicAdherenceScore(llm=llm)
    sample = build_sample()
    try:
        score = metric.multi_turn_score(sample)
        print("TopicAdherenceScore:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
