"""Minimal RemoteGraph example for the `router_graph` subgraph.

Run with uv (server must be running separately):

    uv run AWARE/agent-evals/router_graph.py \
        --url http://127.0.0.1:2024 \
        --user_input "Call rag, then react, then reasoner to find the answer."

Launch the LangGraph dev server for the router subgraph (from repo root):

    uv run langgraph dev --config langgraph_server/router.json

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
from typing import Any, cast

from langgraph.pregel.remote import RemoteGraph


def _payload(user_input: str) -> dict[str, Any]:
    """Return a schema-valid input payload for `router_graph`.

    The payload matches the input demonstrated in the router subgraph's
    builder/tests. Adjust `user_input` to try different routing prompts.
    """
    return {"user_input": user_input}


async def _amain() -> None:
    """Parse args, invoke the remote `router_graph`, and print the result."""
    parser = argparse.ArgumentParser(
        description=(
            "Invoke a running LangGraph dev server's router_graph using RemoteGraph"
        )
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:2024",
        help="Base URL of the running LangGraph server",
    )
    parser.add_argument(
        "--user_input",
        default=(
            "Call rag, then call react and finally call the reasoner to find the"
            " answer."
        ),
        help="User input for the router graph",
    )
    args = parser.parse_args()

    graph = RemoteGraph("router_graph", url=args.url)  # type: ignore[arg-type]
    payload = _payload(args.user_input)

    print(f"Invoking 'router_graph' at {args.url} with payload:\n{payload}\n")

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
