"""LangSmith pytest integration for the receptionist graph.

This test runs the receptionist subgraph, logs inputs/outputs to LangSmith,
and evaluates trajectories using AgentEvals.

Environment variables required for full execution:
- OPENAI_API_KEY (for both the graph and LLM-as-judge)
- LANGSMITH_API_KEY (for LangSmith logging)
- LANGSMITH_TRACING=true (to enable tracing)

Run:

uv run -m pytest -q tests/test_receptionist_langsmith.py --langsmith-output
"""

# %%
from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from langsmith import testing as t

from src.graphs.receptionist_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer,
)
from src.graphs.receptionist_subgraph.schemas import ReceptionistOutputSchema


def _build_reference_messages(user_prompt: str) -> list[dict[str, str]]:
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


async def _run_graph_once(
    *,
    user_prompt: str | None = None,
    initial_messages: list[Any] | None = None,
    initial_schema: ReceptionistOutputSchema | dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Run the graph and return OpenAI-style messages for evaluation.

    Accepts optional structured history and an initial schema to simulate
    context injection or partial profiles.
    """
    steps: list[str] = ["__start__"]
    last_schema: ReceptionistOutputSchema | None = None

    if initial_messages is None:
        if user_prompt is None:
            raise ValueError("Provide user_prompt or initial_messages")
        initial_messages = [user_prompt]

    init_state: dict[str, Any] = {"messages": initial_messages}
    if initial_schema is not None:
        init_state["receptionist_output_schema"] = initial_schema

    config: dict[str, Any] = {"configurable": {"thread_id": uuid.uuid4()}}
    async for update in graph_with_in_memory_checkpointer.astream(
        init_state,
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

    assistant_text = (
        last_schema.direct_response_to_the_user if last_schema else None
    ) or (
        "To help you best, please share your name, address, employment "
        "status, last job details, and job preferences."
    )

    first_user = next(
        (m for m in initial_messages if isinstance(m, str)),
        user_prompt or "",
    )
    return [
        {"role": "user", "content": str(first_user)},
        {"role": "assistant", "content": assistant_text},
    ]


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_receptionist_trajectory_accuracy() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping LLM-as-judge evaluation")

    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            create_trajectory_llm_as_judge,
        )
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"agentevals unavailable: {exc}")

    user_prompt = "Hi, I'm looking for an entry-level retail job."
    outputs = await _run_graph_once(user_prompt=user_prompt)
    reference_outputs = _build_reference_messages(user_prompt)

    # Log to LangSmith
    t.log_inputs({})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    model = os.environ.get("AGENTEVALS_MODEL", "openai:o3-mini")
    evaluator = create_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT
    )
    evaluator(outputs=outputs, reference_outputs=reference_outputs)


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_receptionist_llm_judge_with_reference_prompt() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping LLM-as-judge evaluation")

    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
            create_trajectory_llm_as_judge,
        )
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"agentevals unavailable: {exc}")

    user_prompt = "I'm seeking part-time retail work in Rockville, MD."
    outputs = await _run_graph_once(user_prompt=user_prompt)
    reference_outputs = _build_reference_messages(user_prompt)

    t.log_inputs({"scenario": "reference_prompt"})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    model = os.environ.get("AGENTEVALS_MODEL", "openai:o3-mini")
    evaluator = create_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE
    )
    evaluator(outputs=outputs, reference_outputs=reference_outputs)


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_receptionist_llm_judge_async_variant() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping async LLM-as-judge evaluation")

    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            create_async_trajectory_llm_as_judge,
        )
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"agentevals unavailable: {exc}")

    user_prompt = "Any cashier roles near Arlington for evenings?"
    outputs = await _run_graph_once(user_prompt=user_prompt)
    reference_outputs = _build_reference_messages(user_prompt)

    t.log_inputs({"scenario": "async_judge"})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    model = os.environ.get("AGENTEVALS_MODEL", "openai:o3-mini")
    evaluator = create_async_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT
    )
    await evaluator(outputs=outputs, reference_outputs=reference_outputs)


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize("feedback_key", ["traj_acc_a", "traj_acc_b", "traj_acc_c"])
async def test_receptionist_llm_judge_feedback_keys(feedback_key: str) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping LLM-as-judge evaluation")

    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            create_trajectory_llm_as_judge,
        )
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"agentevals unavailable: {exc}")

    user_prompt = "Weekend supermarket jobs around DC?"
    outputs = await _run_graph_once(user_prompt=user_prompt)
    reference_outputs = _build_reference_messages(user_prompt)

    t.log_inputs({"scenario": "feedback_keys", "feedback_key": feedback_key})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    model = os.environ.get("AGENTEVALS_MODEL", "openai:o3-mini")
    evaluator = create_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT, feedback_key=feedback_key
    )
    evaluator(outputs=outputs, reference_outputs=reference_outputs)


@pytest.mark.langsmith
@pytest.mark.asyncio
async def test_receptionist_trajectory_strict_match() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping trajectory match evaluation")

    try:
        from agentevals.trajectory.match import create_trajectory_match_evaluator
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"agentevals unavailable: {exc}")

    user_prompt = "Hi, I'm looking for an entry-level retail job."
    outputs = await _run_graph_once(user_prompt=user_prompt)
    reference_outputs = _build_reference_messages(user_prompt)

    # Log to LangSmith
    t.log_inputs({})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    evaluator = create_trajectory_match_evaluator()
    evaluator(
        outputs={"messages": outputs},
        reference_outputs={"messages": reference_outputs},
    )


# --- Extended scenarios for robust statistics and LangSmith logging ---

VARIED_PROMPTS = [
    "Hi, I'm looking for an entry-level retail job.",
    "I'm seeking part-time retail work in Rockville, MD.",
    "Any cashier roles near Arlington for evenings?",
    "Weekend supermarket jobs around DC?",
    "Store associate roles with flexible schedule in Fairfax.",
]


def _build_structured_history() -> list[Any]:
    return [
        "I'm interested in retail jobs.",
        {"human": "Previously, I mentioned I'm in Arlington."},
        {"ai": "What is your name and current address?"},
        "Can you help me find entry-level roles?",
    ]


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", VARIED_PROMPTS)
@pytest.mark.parametrize("rep", [0, 1, 2])
async def test_varied_prompts_llm_judge_and_match(prompt: str, rep: int) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping evaluators")

    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            create_trajectory_llm_as_judge,
        )
        from agentevals.trajectory.match import create_trajectory_match_evaluator
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"agentevals unavailable: {exc}")

    outputs = await _run_graph_once(user_prompt=prompt)
    reference_outputs = _build_reference_messages(prompt)

    t.log_inputs({"scenario": "varied_prompts", "rep": rep, "prompt": prompt})
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    model = os.environ.get("AGENTEVALS_MODEL", "openai:o3-mini")
    llm_eval = create_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT
    )
    llm_eval(outputs=outputs, reference_outputs=reference_outputs)

    match_eval = create_trajectory_match_evaluator()
    match_eval(
        outputs={"messages": outputs},
        reference_outputs={"messages": reference_outputs},
    )


@pytest.mark.langsmith
@pytest.mark.asyncio
@pytest.mark.parametrize("rep", [0, 1, 2])
async def test_history_and_partial_profile(rep: int) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping evaluators")

    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            create_trajectory_llm_as_judge,
        )
        from agentevals.trajectory.match import create_trajectory_match_evaluator
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"agentevals unavailable: {exc}")

    initial_messages = _build_structured_history()
    initial_schema = ReceptionistOutputSchema(
        user_name="John Doe",
        direct_response_to_the_user=None,
    )
    outputs = await _run_graph_once(
        initial_messages=initial_messages, initial_schema=initial_schema
    )
    reference_outputs = _build_reference_messages(str(initial_messages[0]))

    t.log_inputs(
        {
            "scenario": "history_and_partial_profile",
            "rep": rep,
        }
    )
    t.log_outputs({"messages": outputs})
    t.log_reference_outputs({"messages": reference_outputs})

    model = os.environ.get("AGENTEVALS_MODEL", "openai:o3-mini")
    llm_eval = create_trajectory_llm_as_judge(
        model=model, prompt=TRAJECTORY_ACCURACY_PROMPT
    )
    llm_eval(outputs=outputs, reference_outputs=reference_outputs)

    match_eval = create_trajectory_match_evaluator()
    match_eval(
        outputs={"messages": outputs},
        reference_outputs={"messages": reference_outputs},
    )


if __name__ == "__main__":
    # Allow running this file directly
    raise SystemExit(
        pytest.main(["-q", __file__, "--langsmith-output"])  # type: ignore[arg-type]
    )
