"""Minimal RemoteGraph example for the `planner_executor_graph` subgraph.

Run with uv (server must be running separately):

    uv run AWARE/agent-evals/planner_executor_graph.py \
        --url http://127.0.0.1:2024 \
        --handoff_input "Call rag, then react, then reasoner to find the answer."

Launch the LangGraph dev server for the planner/executor subgraph (from root):

    uv run langgraph dev --config langgraph_server/planner_executor.json

See Also:
- RemoteGraph reference:
  https://langchain-ai.github.io/langgraph/reference/remote_graph/
- Use a remote graph:
  https://docs.langchain.com/langgraph-platform/use-remote-graph
"""

# %%
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, cast

from langgraph.pregel.remote import RemoteGraph
from ragas import SingleTurnSample
from ragas.metrics import NonLLMStringSimilarity


FEWSHOTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "graphs"
    / "planner_executor_subgraph"
    / "fewshots.yml"
)


def _payload(handoff_input: str) -> dict[str, Any]:
    """Return a schema-valid payload for `planner_executor_graph`.

    The payload matches inputs demonstrated in the planner/executor subgraph
    tests. Adjust `handoff_input` to drive different planner instructions.
    """
    return {"handoff_input": handoff_input}


def _load_fewshot_reference(user_input: str) -> tuple[str | None, str | None, str]:
    """Load fewshots and return (reference_output, reference_input, selection).

    Tries to match by exact `input`. If no match found, falls back to the first
    example pair and indicates selection reason for logging.
    """
    try:
        import yaml  # Local import to avoid hard dep if not needed
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not import yaml: {exc}")
        return None, None, "yaml_import_failed"

    if not FEWSHOTS_PATH.exists():
        print(f"[WARN] Fewshots file not found: {FEWSHOTS_PATH}")
        return None, None, "fewshots_missing"

    data = yaml.safe_load(FEWSHOTS_PATH.read_text(encoding="utf-8"))
    shots = data.get("FEW_SHOTS_PLANNER", [])

    # The structure alternates input/output blocks in the provided file.
    # We'll walk pairs: [{input: ...}, {output: ...}]
    for i in range(0, len(shots), 2):
        block_in = shots[i]
        block_out = shots[i + 1] if i + 1 < len(shots) else None
        if not (isinstance(block_in, dict) and isinstance(block_out, dict)):
            continue
        if str(block_in.get("input", "")).strip() == user_input.strip():
            output_text = block_out.get("output")
            if isinstance(output_text, str):
                return output_text, str(block_in.get("input", "")), "matched_by_input"

    # Fallback: use the first valid pair if available
    for i in range(0, len(shots), 2):
        block_in = shots[i]
        block_out = shots[i + 1] if i + 1 < len(shots) else None
        if not (isinstance(block_in, dict) and isinstance(block_out, dict)):
            continue
        output_text = block_out.get("output")
        if isinstance(output_text, str):
            return output_text, str(block_in.get("input", "")), "first_example_fallback"

    return None, None, "no_valid_pairs"


def _extract_step_sequence(plan: dict[str, Any]) -> list[str]:
    """Extract the ordered list of suggested_subgraph values from plan steps."""
    steps = plan.get("steps") if isinstance(plan, dict) else None
    if not isinstance(steps, list):
        return []
    sequence: list[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        subgraph = step.get("suggested_subgraph")
        if isinstance(subgraph, str):
            sequence.append(subgraph)
    return sequence


def _format_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


async def _amain() -> None:
    """Parse args, invoke the remote `planner_executor_graph`, print result.

    Also evaluates the generated plan against the project fewshots using two
    checks:
      1) NonLLMStringSimilarity between the pretty-printed plan and reference
         output text (if available).
      2) Ordered subgraph sequence match (exact) if both sequences are present.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Invoke a running LangGraph dev server's planner_executor_graph using"
            " RemoteGraph and evaluate the plan against fewshots."
        )
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:2024",
        help="Base URL of the running LangGraph server",
    )
    parser.add_argument(
        "--handoff_input",
        default=(
            "Call rag, then call react and finally call the reasoner to find the"
            " answer."
        ),
        help="Instruction the planner should process",
    )
    args = parser.parse_args()

    graph = RemoteGraph("planner_executor_graph", url=args.url)  # type: ignore
    payload = _payload(args.handoff_input)

    print(f"Invoking 'planner_executor_graph' at {args.url} with payload:\n{payload}\n")

    try:
        fn_async = graph.ainvoke
    except AttributeError:
        fn_async = None

    if fn_async is not None:
        result: Any = await cast(Any, fn_async)(payload)
    else:
        try:
            fn_sync = graph.invoke
        except AttributeError:
            fn_sync = None
        if fn_sync is not None:
            result = cast(Any, fn_sync)(payload)
        else:
            msg = "Remote graph has neither 'ainvoke' nor 'invoke' methods"
            raise RuntimeError(msg)

    print("Raw result:")
    print(_format_json(result))
    print()

    # ===== Evaluation =====
    # 1) String similarity against fewshot reference (if available)
    ref_text, ref_input_used, selection = _load_fewshot_reference(args.handoff_input)
    similarity_score: float | None = None

    if ref_text is not None:
        metric = NonLLMStringSimilarity()
        # Compare pretty plan JSON with the fewshot reference text
        generated_plan_text = _format_json(result.get("plan", {}))
        sample = SingleTurnSample(
            user_input=args.handoff_input,
            response=generated_plan_text,
            reference=ref_text,
            retrieved_contexts=[],
        )
        similarity_score = metric.single_turn_score(sample)

    # 2) Sequence order exact match if we can derive both
    generated_seq = _extract_step_sequence(result.get("plan", {}))
    expected_seq: list[str] | None = None
    if ref_text:
        # Heuristic: find suggested_subgraph: <name> lines in the reference
        expected_seq = []
        for line in ref_text.splitlines():
            line_stripped = line.strip()
            if line_stripped.lower().startswith("suggested_subgraph:"):
                _, rhs = line_stripped.split(":", 1)
                expected_seq.append(rhs.strip())
        if not expected_seq:
            expected_seq = None

    order_match: bool | None = None
    if expected_seq is not None:
        order_match = generated_seq == expected_seq

    # Print a rich, well-organized report
    print("Evaluation Report:")
    print("- Input:")
    print(f"  {args.handoff_input}")
    print("- Generated plan (steps -> suggested_subgraph):")
    for idx, name in enumerate(generated_seq, start=1):
        print(f"  {idx}. {name}")

    if expected_seq is not None:
        print("- Expected sequence (from fewshots):")
        for idx, name in enumerate(expected_seq, start=1):
            print(f"  {idx}. {name}")
    else:
        print("- Expected sequence: not available (no match in fewshots)")

    if similarity_score is not None:
        print(f"- NonLLMStringSimilarity (plan vs. reference): {similarity_score:.4f}")
    else:
        print("- NonLLMStringSimilarity: skipped (no fewshot reference found)")

    if order_match is not None:
        print(f"- Step order exact match: {order_match}")
    else:
        print("- Step order exact match: skipped (no reference sequence)")

    print()
    print("Result (compact):")
    print(
        json.dumps(
            {
                "similarity_score": similarity_score,
                "generated_sequence": generated_seq,
                "expected_sequence": expected_seq,
                "order_match": order_match,
                "reference_input_used": ref_input_used,
                "reference_selection": selection,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(_amain())
