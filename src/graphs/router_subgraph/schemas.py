"""Router submodule for the workflow graph."""

# %%
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


with Path("src/graphs/router_subgraph/handoff-subgraphs.yml").open(
    encoding="utf-8"
) as f:
    subgraphs = yaml.safe_load(f)
    SUBGRAPH_NAMES = tuple(
        subgraph["name"] for subgraph in subgraphs["handoff_subgraphs"]
    )

NextSubgraph = Literal[*SUBGRAPH_NAMES]


class RouterOutputSchema(BaseModel):
    """Router output schema."""

    direct_response_to_the_user: str | None = Field(
        default=None,
        description=(
            "The direct response to a user trivial question. "
            "When NO sub-graph or NO tools are needed. "
            "If none, the next_subgraph is used."
        ),
    )
    next_subgraph: NextSubgraph | None = Field(
        default=None,
        description=(
            "The next subgraph to be used. "
            "If none, the direct_response_to_the_user is used."
        ),
    )

    @property
    def correctness(self) -> bool:
        """Check if the response is correct."""
        return (self.direct_response_to_the_user is not None) != (
            self.next_subgraph is not None
        )


class RouterSubgraphState(MessagesState):
    """Router subgraph state."""

    user_input: str = Field(default="")
    router_output: RouterOutputSchema = Field(default_factory=RouterOutputSchema)
    subgraph_name: NextSubgraph | None = Field(default=None)
    fallback_reason: str | None = Field(default=None)


if __name__ == "__main__":
    with Path("src/graphs/router_subgraph/fewshots.yml").open(encoding="utf-8") as f:
        fewshots = yaml.safe_load(f)

    for example in fewshots["Routing_examples"]:
        output = example["output"]
        router_output = RouterOutputSchema(**output)
        print(
            f"Input: '{example['input']}'\n"
            f"Output: {output}\n"
            f"Correctness: {router_output.correctness}\n"
            "---"
        )

# %%
