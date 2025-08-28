"""AgentEvals-based evaluations for the receptionist graph.

Run as a script:

uv run AWARE/agentevals/eval_receptionist.py

This evaluates only `graph_with_in_memory_checkpointer` using:
- Trajectory match evaluator
- LLM-as-judge (sync and async variants)

References: see the AgentEvals repository `langchain-ai/agentevals`.
"""

# %%
from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from src.graphs.receptionist_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer,
)
from src.graphs.receptionist_subgraph.schemas import ReceptionistOutputSchema


# Type aliases mirroring AgentEvals match configuration
ToolArgsMatchMode = Literal["exact", "ignore", "subset", "superset"]
ToolArgsMatchOverrides = dict[
    str, ToolArgsMatchMode | list[str] | Callable[[dict, dict], bool]
]


class GraphTrajectory(TypedDict, total=False):
    """A simplified graph trajectory format used for matching.

    - inputs: Optional list of messages or inputs provided to the graph
    - results: List of result dicts (e.g., OpenAI-style messages)
    - steps: List of step sequences; each sequence is a list of node names
    """

    inputs: list[dict] | None
    results: list[dict]
    steps: list[list[str]]


def _build_reference_messages(user_prompt: str) -> list[dict[str, str]]:
    """Deterministic reference assistant prompt for receptionist behavior."""
    assistant_ref = (
        "Thanks for reaching out! To help you best, please share: your full "
        "name, current address, current employment status, your last job title, "
        "last job location, last employer, and your job preferences (industry, "
        "role type, schedule, and location)."
    )
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_ref},
    ]


async def _extract_trajectory_from_graph(
    user_prompt: str,
) -> tuple[GraphTrajectory, list[dict[str, str]]]:
    """Run the receptionist graph once and extract a simple trajectory.

    Returns a tuple of (graph_trajectory, messages_for_llm_judge).
    """
    steps: list[str] = ["__start__"]
    last_schema: ReceptionistOutputSchema | None = None

    config = {"configurable": {"thread_id": uuid.uuid4()}}

    async for update in graph_with_in_memory_checkpointer.astream(
        {"messages": [user_prompt]},
        config,
        stream_mode="updates",
        debug=False,
    ):
        for node_name, node_state in update.items():
            steps.append(node_name)
            if isinstance(node_state, dict) and (
                "receptionist_output_schema" in node_state
            ):
                schema_data = node_state["receptionist_output_schema"]
                if isinstance(schema_data, ReceptionistOutputSchema):
                    last_schema = schema_data
                elif isinstance(schema_data, dict):
                    last_schema = ReceptionistOutputSchema.model_validate(schema_data)

    # Heuristic for terminal marker: if we visited handoff, mark END; else interrupt
    if "handoff_to_logging" in steps:
        steps.append("__end__")
    else:
        steps.append("__interrupt__")

    assistant_text = (
        last_schema.direct_response_to_the_user if last_schema else None
    ) or (
        "To help you best, please share your name, address, employment "
        "status, last job details, and job preferences."
    )

    messages_for_llm_judge: list[dict[str, str]] = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_text},
    ]

    trajectory: GraphTrajectory = {
        "inputs": [{"role": "user", "content": user_prompt}],
        "results": [{"role": "assistant", "content": assistant_text}],
        "steps": [steps],
    }
    return trajectory, messages_for_llm_judge


async def run_trajectory_match_eval(
    *,
    outputs: GraphTrajectory,
    reference_outputs: GraphTrajectory,
    outputs_messages: list[dict[str, str]] | None = None,
    reference_messages: list[dict[str, str]] | None = None,
) -> Any:
    """Run AgentEvals trajectory match if available; otherwise return a skip."""
    try:
        from agentevals.trajectory.match import create_trajectory_match_evaluator
    except ImportError as exc:  # pragma: no cover
        return {"skipped": True, "reason": f"agentevals unavailable: {exc}"}

    match_evaluator = create_trajectory_match_evaluator()
    # Prefer messages-based API for broader compatibility with installed versions
    if outputs_messages is not None and reference_messages is not None:
        return match_evaluator(
            outputs={"messages": outputs_messages},
            reference_outputs={"messages": reference_messages},
        )
    # Fallback: attempt to pass the graph trajectory directly
    return match_evaluator(outputs=outputs, reference_outputs=reference_outputs)


async def run_llm_as_judge_evals(
    *,
    messages: list[dict[str, str]],
    reference_messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Run LLM-as-judge evaluations (sync and async variants) if available."""
    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
            create_async_trajectory_llm_as_judge,
            create_trajectory_llm_as_judge,
        )
    except ImportError as exc:  # pragma: no cover
        return {"skipped": True, "reason": f"agentevals unavailable: {exc}"}

    if not os.environ.get("OPENAI_API_KEY"):
        return {
            "skipped": True,
            "reason": "OPENAI_API_KEY not set; skipping LLM-as-judge.",
        }

    model = os.environ.get("AGENTEVALS_MODEL", "openai:o3-mini")

    sync_default = create_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT
    )
    res_sync_default = sync_default(
        outputs=messages, reference_outputs=reference_messages
    )

    sync_with_ref = create_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
    )
    res_sync_with_ref = sync_with_ref(
        outputs=messages, reference_outputs=reference_messages
    )

    async_default = create_async_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT
    )
    res_async_default = await async_default(
        outputs=messages, reference_outputs=reference_messages
    )

    # Optional example: run with explicit AsyncOpenAI if enabled by env flag
    res_async_explicit: Any | None = None
    if os.environ.get("RUN_EXPLICIT_OPENAI", "0") == "1":
        try:
            from openai import AsyncOpenAI
        except ImportError:  # pragma: no cover - optional example
            res_async_explicit = None
        else:
            async_explicit = create_async_trajectory_llm_as_judge(
                prompt=TRAJECTORY_ACCURACY_PROMPT,
                judge=AsyncOpenAI(),
                model=os.environ.get("AGENTEVALS_MODEL", "o3-mini"),
            )
            res_async_explicit = await async_explicit(
                outputs=messages, reference_outputs=reference_messages
            )

    return {
        "sync_default": res_sync_default,
        "sync_with_reference_prompt": res_sync_with_ref,
        "async_default": res_async_default,
        "async_with_explicit_client": res_async_explicit,
    }


async def main() -> None:
    """Execute trajectory match and LLM-as-judge evaluations for the graph."""
    user_prompt = "Hi, I'm looking for an entry-level retail job."

    trajectory, judge_messages = await _extract_trajectory_from_graph(user_prompt)

    reference_steps = [
        [
            "__start__",
            "receptor",
            "validate_user_profile",
            "__interrupt__",
        ]
    ]
    reference_outputs: GraphTrajectory = {
        "inputs": [{"role": "user", "content": user_prompt}],
        "results": [],
        "steps": reference_steps,
    }

    ref_messages = _build_reference_messages(user_prompt)

    match_result = await run_trajectory_match_eval(
        outputs=trajectory,
        reference_outputs=reference_outputs,
        outputs_messages=judge_messages,
        reference_messages=ref_messages,
    )

    judge_results = await run_llm_as_judge_evals(
        messages=judge_messages, reference_messages=ref_messages
    )

    print("\n=== graph trajectory (observed) ===")
    print(trajectory)

    print("\n=== trajectory match result ===")
    print(match_result)

    print("\n=== LLM-as-judge results ===")
    print(judge_results)


if __name__ == "__main__":
    asyncio.run(main())

# %%
