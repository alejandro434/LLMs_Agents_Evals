"""AgentGoalAccuracyWithoutReference example for Ragas.

What this measures
    Whether the agent accomplished the inferred user goal without requiring
    a human-annotated reference. The goal is derived from the dialog.

When to use
    - Scalable system-level evaluation when references are costly to create.
    - Production monitoring to estimate task success rates.

Trade-offs
    - LLM inference of goals can be ambiguous. Use with representative
      samples and consider spot-checking.

Implementation notes
    - Multi-turn metric; ensure conversations contain clear signals of
      success (acknowledgments, confirmations, state changes).

Run
    uv run docs/ragas/examples/agent_goal_accuracy_noref.py
"""

# %%
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from ragas import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.messages import AIMessage, HumanMessage
from ragas.metrics import AgentGoalAccuracyWithoutReference


def build_sample() -> MultiTurnSample:
    """Create a simple conversation where the implicit goal is achieved."""
    messages = [
        HumanMessage(content="Set a reminder to call mom at 6 PM today."),
        AIMessage(content="Setting reminder for 6 PM today."),
        AIMessage(content="Reminder set to call mom at 6 PM."),
    ]
    return MultiTurnSample(
        user_input=messages,
    )


def main() -> None:
    """Run AgentGoalAccuracyWithoutReference with an LLM if available."""
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping LLM example: OPENAI_API_KEY not set.")
        return
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    metric = AgentGoalAccuracyWithoutReference(llm=llm)
    sample = build_sample()
    try:
        score = metric.multi_turn_score(sample)
        print("AgentGoalAccuracyWithoutReference:", score)
    except Exception as e:
        print("Skipping LLM example due to error:", e)


if __name__ == "__main__":
    main()
