"""AgentGoalAccuracyWithReference example for Ragas.

What this measures
    Whether the agent achieved the user's objective, evaluated against a
    human-authored `reference` description of a successful outcome.

When to use
    - Planner/executor workflows with clear success criteria.
    - System-level end-to-end validation for goal completion.

Trade-offs
    - Requires a high-quality reference; LLM judge introduces cost.
    - Great for dashboards, but less detailed than step-level checks.

Implementation notes
    - Multi-turn metric; design concise and unambiguous references.

Run
    uv run docs/ragas/examples/agent_goal_accuracy_ref.py
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics import AgentGoalAccuracyWithReference


def build_sample() -> MultiTurnSample:
    messages = [
        HumanMessage(content="Please book a flight from JFK to LAX tomorrow."),
        AIMessage(content="Searching available flights."),
        AIMessage(
            content="A flight at 10:00 AM has been booked from JFK to LAX for tomorrow."
        ),
    ]
    return MultiTurnSample(
        user_input=messages,
        reference=("A flight was successfully booked from JFK to LAX for tomorrow."),
    )


def main() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    metric = AgentGoalAccuracyWithReference(llm=llm)
    sample = build_sample()
    try:
        score = metric.multi_turn_score(sample)
        print("AgentGoalAccuracyWithReference:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
