"""Deterministic StringPresence example for Ragas.

What this measures
    StringPresence checks if `reference` is a substring of `response`,
    returning 1.0 when present and 0.0 otherwise.

When to use
    - Keyword/constraint checks in generated answers (e.g., must mention
      "privacy policy").
    - Guardrails: ensure the model includes a mandatory disclaimer.
    - AWARE: quick validation of routing or policy-compliance strings.

Trade-offs
    - Surface-level: does not validate correctness or semantics.
    - Can be brittle if phrasing varies; consider semantic metrics when
      wording is flexible.

Implementation notes
    - Pure string operation (no LLMs). Reproducible and fast.

Run
    uv run docs/ragas/examples/string_presence.py
"""

# %%
from __future__ import annotations

from ragas import SingleTurnSample
from ragas.metrics import StringPresence


def build_sample() -> SingleTurnSample:
    """Create a simple single-turn sample.

    StringPresence checks if `reference` is a substring of `response`.
    """
    return SingleTurnSample(
        user_input="Summarize Python's purpose",
        response="Python is a high-level programming language used for general-purpose programming.",
        reference="general-purpose programming",
        retrieved_contexts=[],
    )


def main() -> None:
    metric = StringPresence()
    sample = build_sample()
    score = metric.single_turn_score(sample)
    print("StringPresence score:", score)


if __name__ == "__main__":
    main()
