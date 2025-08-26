"""Router submodule for the workflow graph."""

# %%
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class ReceptionistOutputSchema(BaseModel):
    """Receptionist output schema."""

    direct_response_to_the_user: str | None = Field(
        default=None,
        description=(
            "A direct response to the user's question, to be used when no handoff is required. "
            "If this field is null, a handoff is necessary."
        ),
    )
    user_name: str | None = Field(
        default=None,
        description="The name of the user.",
    )

    user_current_address: str | None = Field(
        default=None,
        description="The user's current residential address.",
    )
    user_employment_status: (
        Literal["employed", "unemployed", "self-employed", "retired"] | None
    ) = Field(
        default=None,
        description="The employment status of the user.",
    )
    user_last_job: str | None = Field(
        default=None,
        description="A description of the user's last job.",
    )
    user_last_job_location: str | None = Field(
        default=None,
        description="The location of the user's last job.",
    )
    user_last_job_company: str | None = Field(
        default=None,
        description="The company of the user's last job.",
    )
    user_job_preferences: str | None = Field(
        default=None,
        description="A summary of the user's career interests, needs, and preferences (e.g., industry, role type, salary expectations, location).",
    )

    handoff_needed: bool = Field(
        default=False,
        description=(
            "Whether a handoff to a human or another subgraph is needed. "
            "If true, the handoff_subgraph is used. "
            "If false, the direct_response_to_the_user is used."
        ),
    )

    @property
    def is_valid_state(self) -> bool:
        """Checks.

        for a valid output state: either a direct response exists
        or a handoff is needed, but not both.
        """
        return (self.direct_response_to_the_user is not None) != self.handoff_needed


class UserProfileSchema(BaseModel):
    """Holds all information related to a user's profile."""

    name: str | None = Field(default=None, description="The user's name.")
    current_address: str | None = Field(
        default=None, description="The user's current residential address."
    )
    employment_status: (
        Literal["employed", "unemployed", "self-employed", "retired"] | None
    ) = Field(default=None, description="The user's current employment status.")
    last_job: str | None = Field(
        default=None,
        description="A description of the user's most recent job title or role.",
    )
    last_job_location: str | None = Field(
        default=None,
        description="The location (e.g., city, state) of the user's last job.",
    )
    last_job_company: str | None = Field(
        default=None, description="The name of the company where the user last worked."
    )
    job_preferences: str | None = Field(
        default=None,
        description="A summary of the user's career interests, needs, and preferences (e.g., industry, role type, salary expectations).",
    )


class ReceptionistSubgraphState(MessagesState):
    """Receptionist subgraph state."""

    receptionist_output_schema: ReceptionistOutputSchema = Field(
        default_factory=ReceptionistOutputSchema
    )
    handoff_needed: bool = Field(default=False)
    user_profile_schema: UserProfileSchema = Field(default_factory=UserProfileSchema)
    fallback_message: str | None = Field(default=None)


if __name__ == "__main__":
    with Path("src/graphs/receptionist_subgraph/fewshots.yml").open(
        encoding="utf-8"
    ) as f:
        fewshots = yaml.safe_load(f)

    for example in fewshots["Receptionist_examples"]:
        output = example["output"]
        receptionist_output = ReceptionistOutputSchema(**output)
        print(
            f"Input: '{example['input']}'\n"
            f"Output: {output}\n"
            f"Correctness: {receptionist_output.correctness}\n"
            "---"
        )

# %%
