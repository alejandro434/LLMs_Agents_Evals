"""ToolCallAccuracy example for Ragas.

What this measures
    Correctness of tool usage in a conversation: tool names, argument
    structures, and sequence compared against `reference_tool_calls`.

When to use
    - ReAct/agentic systems using external tools or APIs.
    - CI checks for regressions in tool selection or ordering.

Trade-offs
    - Defaults to exact matching for arguments; can plug custom comparators
      (e.g., `NonLLMStringSimilarity`) for flexible natural-language args.

Implementation notes
    - Works with `MultiTurnSample` and `ragas.messages` types.

Run
    uv run docs/ragas/examples/tool_call_accuracy.py
"""

# %%
from __future__ import annotations

from ragas import MultiTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from ragas.metrics import ToolCallAccuracy


def build_sample() -> MultiTurnSample:
    messages = [
        HumanMessage(content="What's the weather in New York?"),
        AIMessage(
            content="Fetching current weather.",
            tool_calls=[ToolCall(name="weather_check", args={"location": "New York"})],
        ),
        ToolMessage(content="It is 75°F and partly cloudy."),
        HumanMessage(content="Convert to Celsius."),
        AIMessage(
            content="Converting to Celsius.",
            tool_calls=[
                ToolCall(
                    name="temperature_conversion", args={"temperature_fahrenheit": 75}
                )
            ],
        ),
        ToolMessage(content="75°F is approximately 23.9°C."),
        AIMessage(content="It is about 23.9°C in New York."),
    ]
    return MultiTurnSample(
        user_input=messages,
        reference_tool_calls=[
            ToolCall(name="weather_check", args={"location": "New York"}),
            ToolCall(
                name="temperature_conversion", args={"temperature_fahrenheit": 75}
            ),
        ],
    )


def main() -> None:
    metric = ToolCallAccuracy()
    sample = build_sample()
    score = metric.multi_turn_score(sample)
    print("ToolCallAccuracy:", score)


if __name__ == "__main__":
    main()
