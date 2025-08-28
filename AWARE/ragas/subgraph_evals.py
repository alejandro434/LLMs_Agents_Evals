"""Minimal RemoteGraph example for subgraph smoke testing.

Run with uv (server must be running separately):

    uv run AWARE/agent-evals/subgraph_evals.py --graph rag_graph

See Also:
 - RemoteGraph reference:
   https://langchain-ai.github.io/langgraph/reference/remote_graph/
 - Use a remote graph:
   https://docs.langchain.com/langgraph-platform/use-remote-graph
 - JS how-to:
   https://langchain-ai.github.io/langgraphjs/how-tos/use-remote-graph/
"""

# %%
from __future__ import annotations

import argparse
import asyncio
from typing import Any, cast

from langgraph.pregel.remote import RemoteGraph


def _valid_payload_for(graph_name: str) -> dict[str, Any]:
    """Return a schema-valid input payload for the given graph name.

    The payloads match the inputs demonstrated in the corresponding
    `lgraph_builder.py` test blocks for each subgraph.
    """
    if graph_name in {"rag_graph", "react_graph", "reasoner_graph"}:
        return {
            "current_step": {
                "instruction": (
                    "Search for information about Python and then analyze it"
                ),
                "suggested_subgraph": "rag",
                "reasoning": (
                    "First retrieve relevant documents, then extract the answer"
                ),
                "result": "",
                "is_complete": False,
            }
        }

    if graph_name == "planner_executor_graph":
        return {
            "handoff_input": (
                "Call rag, then call react and finally call the reasoner to find "
                "the answer."
            )
        }

    if graph_name == "router_graph":
        return {
            "user_input": (
                "Call rag, then call react and finally call the reasoner to find "
                "the answer."
            )
        }

    raise ValueError(f"Unknown graph name: {graph_name}")


async def _amain() -> None:
    """Parse args, invoke the selected remote graph, and print the result."""
    parser = argparse.ArgumentParser(
        description=(
            "Invoke a running LangGraph dev server using RemoteGraph with a valid "
            "payload for a chosen subgraph."
        )
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:2024",
        help="Base URL of the running LangGraph server",
    )
    parser.add_argument(
        "--graph",
        default="rag_graph",
        choices=[
            "router_graph",
            "planner_executor_graph",
            "rag_graph",
            "react_graph",
            "reasoner_graph",
        ],
        help="Which graph to invoke",
    )
    args = parser.parse_args()

    graph = RemoteGraph(args.graph, url=args.url)  # type: ignore
    payload = _valid_payload_for(args.graph)

    print(f"Invoking '{args.graph}' at {args.url} with payload:\n{payload}\n")
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
    print("Result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(_amain())
