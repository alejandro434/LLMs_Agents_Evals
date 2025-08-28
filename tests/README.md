"""LangSmith-enabled test guide.

This document explains how to prepare, configure, and run the tests that log
results to LangSmith for the receptionist graph.
"""

# %%

## Prerequisites

- Python managed by uv (already configured in the repo)
- Valid API keys and tracing enabled for LangSmith and your LLM provider

Required environment variables:

- `OPENAI_API_KEY` — used by the graph and by AgentEvals LLM-as-judge
- `LANGSMITH_API_KEY` — used for LangSmith logging
- `LANGSMITH_TRACING=true` — enables tracing/logging to LangSmith

Optional environment variables:

- `AGENTEVALS_MODEL` — override the default judge model (`openai:o3-mini`)

## Installing dependencies

Dependencies are managed with `uv`. If you haven’t synced yet:

```bash
uv sync
```

## Running the test suite

Run all tests (including LangSmith logging):

```bash
LANGSMITH_TRACING=true \
LANGSMITH_API_KEY=... \
OPENAI_API_KEY=... \
uv run -m pytest -q --langsmith-output
```

Run only the receptionist LangSmith tests:

```bash
LANGSMITH_TRACING=true \
LANGSMITH_API_KEY=... \
OPENAI_API_KEY=... \
uv run -m pytest -q tests/test_receptionist_langsmith.py --langsmith-output
```

Run a fast subset (LLM-as-judge experiments only):

```bash
LANGSMITH_TRACING=true \
LANGSMITH_API_KEY=... \
OPENAI_API_KEY=... \
uv run -m pytest -q tests/test_receptionist_langsmith.py -k 'llm_judge' --langsmith-output
```

## What gets logged to LangSmith

The tests use the LangSmith pytest integration and log the following per case:

- Inputs: scenario metadata (prompt, repetition, etc.)
- Outputs: OpenAI-style `messages` produced by the receptionist graph
- Reference outputs: OpenAI-style `messages` representing expected behavior

Evaluators used (from AgentEvals):

- LLM-as-judge with `TRAJECTORY_ACCURACY_PROMPT`
- LLM-as-judge with `TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE`
- Async LLM-as-judge variant
- Trajectory strict match on `messages`

## Common commands

- Run a quick smoke of the eval script (no LangSmith integration):

```bash
OPENAI_API_KEY=... uv run AWARE/agentevals/eval_receptionist.py
```

- Run only trajectory/match baseline tests:

```bash
LANGSMITH_TRACING=true \
LANGSMITH_API_KEY=... \
OPENAI_API_KEY=... \
uv run -m pytest -q tests/test_receptionist_langsmith.py -k 'receptionist_trajectory' --langsmith-output
```

## Notes

- Setting `AGENTEVALS_MODEL` lets you switch to another judge model.
- All tests invoke the live `graph_with_in_memory_checkpointer`.
- Make sure to avoid committing secrets; prefer `.env` exports in your shell.

## References

- AgentEvals repo and examples: `https://github.com/langchain-ai/agentevals`
- LangSmith pytest integration: `https://docs.langchain.com/langsmith/pytest`
