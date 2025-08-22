"""Nodes logic.

uv run -m src.graphs.router_subgraph.nodes_logic

"""

# %%
from inspect import getsource
from pathlib import Path
from typing import Literal

import yaml
from langgraph.types import Command

from src.graphs.router_subgraph.chains import chain
from src.graphs.router_subgraph.schemas import RouterOutputSchema, RouterSubgraphState


with Path("src/graphs/router_subgraph/handoff-subgraphs.yml").open(
    encoding="utf-8"
) as f:
    subgraphs = yaml.safe_load(f)
    SUBGRAPH_NAMES = tuple(
        subgraph["name"] for subgraph in subgraphs["handoff_subgraphs"]
    )

NextSubgraph = Literal[*SUBGRAPH_NAMES]


async def router_node(state: RouterSubgraphState) -> Command[NextSubgraph]:
    """Router node."""
    response = await chain.ainvoke(
        state.get("fallback_reason", "") + state["user_input"]
    )
    print(f"Correctness: {response.correctness}")

    if not response.correctness:
        return Command(
            goto="router_node",
            update={
                "fallback_reason": (
                    "You did not pass the correctness check in "
                    + getsource(RouterOutputSchema)
                    + f"Try to answer the following question again: {state['user_input']}."
                ),
            },
        )
    return Command(goto=response.next_subgraph, update={"fallback_reason": None})


if __name__ == "__main__":
    import asyncio

    async def main():
        """Main function."""
        state = RouterSubgraphState(user_input="what is the meaning of life?")
        response = await router_node(state)
        print(response)

    asyncio.run(main())

# %%
