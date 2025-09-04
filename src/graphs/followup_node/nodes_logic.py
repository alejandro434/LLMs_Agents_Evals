"""Nodes logic.

uv run -m src.graphs.followup_node.nodes_logic

"""

# %%

from src.graphs.followup_node.chains import followup_chain
from src.graphs.followup_node.schemas import FollowupOutputSchema, FollowupSubgraphState


async def followup_node(
    state: FollowupSubgraphState,
) -> FollowupOutputSchema:
    """Followup node."""
    response = await followup_chain.ainvoke(state["messages"])

    return response


if __name__ == "__main__":
    import asyncio

    async def main():
        """Main function."""
        user_input = FollowupSubgraphState(messages=["Hi!."])
        response = await followup_node(user_input)
        print(response)

    asyncio.run(main())
