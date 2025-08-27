"""Router submodule for the workflow graph.

uv run -m src.graphs.receptionist_subgraph.schemas
"""

# %%
from __future__ import annotations

from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def _validate_xor_state(self) -> ReceptionistOutputSchema:
        """Enforce XOR between direct response and handoff flags.

        Exactly one of `direct_response_to_the_user` and `handoff_needed` must
        indicate the path: either provide a direct response (non-null) with no
        handoff, or no direct response with a handoff.
        """
        if self.is_valid_state:
            return self
        raise ValueError(
            "Exactly one of 'direct_response_to_the_user' (non-null) or "
            "'handoff_needed' (True) must be set."
        )

    @property
    def user_info_complete(self) -> bool:
        """Return True if all user info fields are present (not None).

        Fields checked:
        - user_name
        - user_current_address
        - user_employment_status
        - user_last_job
        - user_last_job_location
        - user_last_job_company
        - user_job_preferences
        """
        required_fields = (
            self.user_name,
            self.user_current_address,
            self.user_employment_status,
            self.user_last_job,
            self.user_last_job_location,
            self.user_last_job_company,
            self.user_job_preferences,
        )
        return all(value is not None for value in required_fields)


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

    @property
    def user_info_complete(self) -> bool:
        """Checks if the user info is complete."""
        return all(
            value is not None
            for value in (
                self.name,
                self.current_address,
                self.employment_status,
                self.last_job,
                self.last_job_location,
                self.last_job_company,
                self.job_preferences,
            )
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
    # Simple, side-effect-safe tests and demonstrations
    print("Running ReceptionistOutputSchema simple tests...")

    # 1) Valid: direct response present, no handoff
    a = ReceptionistOutputSchema(
        direct_response_to_the_user="Here is your answer.",
        handoff_needed=False,
    )
    if not a.is_valid_state:
        raise AssertionError("Expected valid state when direct response is present")

    # 2) Valid: no direct response, handoff required
    b = ReceptionistOutputSchema(
        direct_response_to_the_user=None,
        handoff_needed=True,
    )
    if not b.is_valid_state:
        raise AssertionError("Expected valid state when handoff is required")

    # 3) Invalid: neither direct response nor handoff
    try:
        ReceptionistOutputSchema(
            direct_response_to_the_user=None,
            handoff_needed=False,
        )
        raise AssertionError("Expected ValueError for invalid XOR state (none)")
    except ValueError:
        pass

    # 4) Invalid: both direct response and handoff
    try:
        ReceptionistOutputSchema(
            direct_response_to_the_user="Text",
            handoff_needed=True,
        )
        raise AssertionError("Expected ValueError for invalid XOR state (both)")
    except ValueError:
        pass

    # 5) user_info_complete should be False by default
    c = ReceptionistOutputSchema(
        direct_response_to_the_user=None,
        handoff_needed=True,
    )
    if c.user_info_complete is not False:
        raise AssertionError("Expected incomplete user info by default")

    # 6) user_info_complete True when all user fields are provided
    d = ReceptionistOutputSchema(
        direct_response_to_the_user=None,
        handoff_needed=True,
        user_name="Jane Doe",
        user_current_address="456 Oak St, Rockville, MD",
        user_employment_status="employed",
        user_last_job="Barista",
        user_last_job_location="Rockville, MD",
        user_last_job_company="Starbucks",
        user_job_preferences=("Full-time admin roles in Montgomery County"),
    )
    if d.user_info_complete is not True:
        raise AssertionError("Expected complete user info when all fields set")

    # 7) Validator message includes helpful hint
    try:
        ReceptionistOutputSchema(
            direct_response_to_the_user=None,
            handoff_needed=False,
        )
    except ValueError as exc:
        if "must be set" not in str(exc):
            raise AssertionError("Expected helpful validator message") from None

    print("ReceptionistOutputSchema tests passed.")

    # UserProfileSchema: user_info_complete property
    print("Running UserProfileSchema simple tests...")

    # 1) Default: incomplete
    up_a = UserProfileSchema()
    if up_a.user_info_complete is not False:
        raise AssertionError("Expected incomplete user profile by default")

    # 2) Fully populated: complete
    up_b = UserProfileSchema(
        name="Jane Doe",
        current_address="456 Oak St, Rockville, MD",
        employment_status="employed",
        last_job="Barista",
        last_job_location="Rockville, MD",
        last_job_company="Starbucks",
        job_preferences="Full-time admin roles in Montgomery County",
    )
    if up_b.user_info_complete is not True:
        raise AssertionError("Expected complete user profile when all fields set")

    # 3) Missing one field: incomplete
    up_c = UserProfileSchema(
        name="Jane Doe",
        current_address="456 Oak St, Rockville, MD",
        employment_status="employed",
        last_job="Barista",
        last_job_location="Rockville, MD",
        last_job_company=None,
        job_preferences="Full-time admin roles in Montgomery County",
    )
    if up_c.user_info_complete is not False:
        raise AssertionError("Expected incomplete user profile when field missing")

    print("UserProfileSchema tests passed.")

# %%
