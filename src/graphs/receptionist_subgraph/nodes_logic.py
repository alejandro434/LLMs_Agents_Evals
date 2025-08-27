"""receptionist node logic.

uv run -m src.graphs.receptionist_subgraph.nodes_logic

"""

# %%
from typing import TYPE_CHECKING, Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from src.graphs.receptionist_subgraph.chains import (
    profiling_chain,
    receptionist_chain,
)


if TYPE_CHECKING:
    from src.graphs.receptionist_subgraph.lgraph_builder import (
        ReceptionistSubgraphState,
    )


async def receptor(
    state: "ReceptionistSubgraphState",
) -> Command[Literal["validate_user_profile"]]:
    """Receptionist node."""
    response = await receptionist_chain.ainvoke(state["messages"])
    return Command(
        goto="validate_user_profile",
        update={"receptionist_output_schema": response},
    )


async def validate_user_profile(
    state: "ReceptionistSubgraphState",
) -> Command[Literal["handoff_to_logging", "receptor"]]:
    """User profile node."""
    if state.get("receptionist_output_schema").user_info_complete:
        return Command(goto="handoff_to_logging")

    user_answer = interrupt(
        state.get("receptionist_output_schema").direct_response_to_the_user
    )
    return Command(goto="receptor", update={"messages": [user_answer]})


async def handoff_to_logging(
    state: "ReceptionistSubgraphState",
) -> Command[Literal[END]]:
    """Handoff to logging node."""
    user_profile = await profiling_chain.ainvoke(
        state.get("receptionist_output_schema")
    )
    return Command(goto=END, update={"user_profile_schema": user_profile})
