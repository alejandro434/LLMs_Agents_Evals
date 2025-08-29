"""Receptionist submodule for the workflow graph.

uv run -m src.graphs.receptionist_subgraph.schemas
"""

# %%
from __future__ import annotations

from typing import Literal

from langgraph.graph import MessagesState
from pydantic import AliasChoices, BaseModel, Field


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
        validation_alias=AliasChoices("user_current_address", "user_address"),
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

    @property
    def at_least_one_user_profile_field_is_filled(self) -> bool:
        """Return True if at least one user profile field is present and filled.

        Checked fields:
        - user_name
        - user_current_address
        - user_employment_status
        - user_last_job
        - user_last_job_location
        - user_last_job_company
        - user_job_preferences

        "Filled" means non-None and, for strings, non-empty after stripping.
        """
        candidates = (
            self.user_name,
            self.user_current_address,
            self.user_employment_status,
            self.user_last_job,
            self.user_last_job_location,
            self.user_last_job_company,
            self.user_job_preferences,
        )
        for value in candidates:
            if isinstance(value, str):
                if value.strip():
                    return True
            elif value is not None:
                return True
        return False

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
        default=None,
        description="The user's current residential address.",
        validation_alias=AliasChoices(
            "current_address", "user_current_address", "user_address"
        ),
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
    def is_valid(self) -> bool:
        """Checks if the user profile is valid (all required fields present).

        This property is used to validate that the profiling chain successfully
        mapped all required fields from ReceptionistOutputSchema.

        Returns:
            bool: True if all required fields are not None, False otherwise.
        """
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

    @property
    def profile_summary(self) -> str:
        """Generate a summary of the user profile for logging/debugging.

        Returns:
            str: A formatted summary of the user profile.
        """
        return (
            f"User: {self.name or 'Unknown'}, "
            f"Location: {self.current_address or 'Unknown'}, "
            f"Status: {self.employment_status or 'Unknown'}, "
            f"Last Job: {self.last_job or 'Unknown'} at {self.last_job_company or 'Unknown'}"
        )


class ReceptionistSubgraphState(MessagesState):
    """Receptionist subgraph state."""

    direct_response_to_the_user: str | None = Field(default=None)

    receptionist_output_schema: ReceptionistOutputSchema = Field(
        default_factory=ReceptionistOutputSchema
    )
    user_profile: UserProfileSchema = Field(default_factory=UserProfileSchema)
    user_request: str | None = Field(default=None)
    selected_agent: Literal["react"] | None = Field(default=None)
    rationale_of_the_handoff: str | None = Field(default=None)

    fallback_message: str | None = Field(default=None)


if __name__ == "__main__":
    # Simple, side-effect-safe tests and demonstrations
    print("Running ReceptionistOutputSchema simple tests...")

    # user_info_complete should be False by default
    c = ReceptionistOutputSchema(
        direct_response_to_the_user=None,
    )
    if c.user_info_complete is not False:
        raise AssertionError("Expected incomplete user info by default")

    # user_info_complete True when all user fields are provided
    d = ReceptionistOutputSchema(
        direct_response_to_the_user=None,
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

    print("ReceptionistOutputSchema tests passed.")

    # UserProfileSchema: is_valid property
    print("Running UserProfileSchema simple tests...")

    # 1) Default: invalid
    up_a = UserProfileSchema()
    if up_a.is_valid is not False:
        raise AssertionError("Expected invalid user profile by default")

    # 2) Fully populated: valid
    up_b = UserProfileSchema(
        name="Jane Doe",
        current_address="456 Oak St, Rockville, MD",
        employment_status="employed",
        last_job="Barista",
        last_job_location="Rockville, MD",
        last_job_company="Starbucks",
        job_preferences="Full-time admin roles in Montgomery County",
    )
    if up_b.is_valid is not True:
        raise AssertionError("Expected valid user profile when all fields set")
    print(f"Profile summary: {up_b.profile_summary}")

    # 3) Missing one field: invalid
    up_c = UserProfileSchema(
        name="Jane Doe",
        current_address="456 Oak St, Rockville, MD",
        employment_status="employed",
        last_job="Barista",
        last_job_location="Rockville, MD",
        last_job_company=None,
        job_preferences="Full-time admin roles in Montgomery County",
    )
    if up_c.is_valid is not False:
        raise AssertionError("Expected invalid user profile when field missing")

    print("UserProfileSchema tests passed.")

    # Pretty print demo using utils
    from src.utils import format_as_json

    example_state = ReceptionistSubgraphState()
    print("\nPretty JSON for default ReceptionistSubgraphState:\n")
    print(format_as_json(example_state))

    # Aliases acceptance tests
    print("\nRunning alias acceptance tests...")
    alias_out = ReceptionistOutputSchema(
        direct_response_to_the_user=None,
        user_address="123 Main St",
    )
    if alias_out.user_current_address != "123 Main St":
        raise AssertionError("Alias 'user_address' not mapped to user_current_address")

    alias_profile = UserProfileSchema(user_address="456 Oak St")
    if alias_profile.current_address != "456 Oak St":
        raise AssertionError("Alias 'user_address' not mapped to current_address")

# %%
