"""receptionist node logic.

uv run -m src.graphs.receptionist_subgraph.nodes_logic

"""

# %%
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from src.graphs.receptionist_subgraph.lgraph_builder import (
        ReceptionistSubgraphState,
    )
from langgraph.graph import END
from langgraph.types import Command, Literal


async def receptionist_node(
    state: "ReceptionistSubgraphState",
) -> Command[Literal[END]]:
    """Receptionist node."""
    return Command(
        [
            {"type": "update", "state": {}},
        ]
    )
