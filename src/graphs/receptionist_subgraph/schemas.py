"""Receptionist submodule for the workflow graph.

uv run -m src.graphs.receptionist_subgraph.schemas
"""

# %%
from __future__ import annotations

from typing import Literal

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
    name: str | None = Field(
        default=None,
        description="The name of the user.",
    )

    current_employment_status: (
        Literal["employed", "unemployed", "self-employed", "retired"] | None
    ) = Field(
        default=None,
        description="The employment status of the user.",
    )

    zip_code: str | None = Field(
        default=None,
        description="The zip code of the user.",
    )
    what_is_the_user_looking_for: str | None = Field(
        default=None,
        description="What the user is looking for.",
    )

    @property
    def at_least_one_user_profile_field_is_filled(self) -> bool:
        """Return True if at least one user profile field is present and filled.

        Checked fields:
        - name
        - current_employment_status
        - zip_code
        - what_is_the_user_looking_for


        "Filled" means non-None and, for strings, non-empty after stripping.
        """
        candidates = (
            self.name,
            self.current_employment_status,
            self.zip_code,
            self.what_is_the_user_looking_for,
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
        - name
        - current_employment_status
        - zip_code
        - what_is_the_user_looking_for
        """
        required_fields = (
            self.name,
            self.current_employment_status,
            self.zip_code,
            self.what_is_the_user_looking_for,
        )
        return all(value is not None for value in required_fields)


class UserProfileSchema(BaseModel):
    """Holds all information related to a user's profile."""

    name: str | None = Field(default=None, description="The user's name.")
    current_employment_status: (
        Literal["employed", "unemployed", "self-employed", "retired"] | None
    ) = Field(default=None, description="The user's current employment status.")
    zip_code: str | None = Field(default=None, description="The user's zip code.")
    what_is_the_user_looking_for: str | None = Field(
        default=None,
        description="What the user is looking for.",
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
                self.current_employment_status,
                self.zip_code,
                self.what_is_the_user_looking_for,
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
            f"Status: {self.current_employment_status or 'Unknown'}, "
            f"Zip Code: {self.zip_code or 'Unknown'}, "
            f"Looking For: {self.what_is_the_user_looking_for or 'Unknown'}"
        )


class ReceptionistSubgraphState(MessagesState):
    """Receptionist subgraph state."""

    direct_response_to_the_user: str | None = Field(default=None)

    receptionist_output_schema: ReceptionistOutputSchema = Field(
        default_factory=ReceptionistOutputSchema
    )
    user_profile: UserProfileSchema = Field(default_factory=UserProfileSchema)
    user_request: str | None = Field(default=None)
    selected_agent: (
        Literal["Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"] | None
    ) = Field(default=None)
    rationale_of_the_handoff: str | None = Field(default=None)

    fallback_message: str | None = Field(default=None)


class UserRequestExtractionSchema(BaseModel):
    """User request extraction schema."""

    task: str = Field(
        description="The user's request written as a task for an specific agent to perform."
    )


if __name__ == "__main__":
    # Simple, side-effect-safe tests and demonstrations
    print("Running ReceptionistOutputSchema simple tests...")

    # user_info_complete should be False by default
    c = ReceptionistOutputSchema(
        direct_response_to_the_user=None,
    )
    if c.user_info_complete is not False:
        raise AssertionError("Expected incomplete user info by default")
    if c.at_least_one_user_profile_field_is_filled is not False:
        raise AssertionError("Expected no filled profile fields by default")

    # user_info_complete True when all user fields are provided
    d = ReceptionistOutputSchema(
        direct_response_to_the_user=None,
        name="Jane Doe",
        current_employment_status="employed",
        zip_code="20850",
        what_is_the_user_looking_for=("Full-time admin roles in Montgomery County"),
    )
    if d.user_info_complete is not True:
        raise AssertionError("Expected complete user info when all fields set")
    if d.at_least_one_user_profile_field_is_filled is not True:
        raise AssertionError("Expected at least one profile field to be filled")

    # at_least_one_user_profile_field_is_filled should ignore whitespace-only
    e = ReceptionistOutputSchema(zip_code="  20850  ")
    if e.at_least_one_user_profile_field_is_filled is not True:
        raise AssertionError(
            "Expected True when a single non-empty trimmed field is provided"
        )

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
        current_employment_status="employed",
        zip_code="20850",
        what_is_the_user_looking_for=("Full-time admin roles in Montgomery County"),
    )
    if up_b.is_valid is not True:
        raise AssertionError("Expected valid user profile when all fields set")
    print(f"Profile summary: {up_b.profile_summary}")

    # 3) Missing one field: invalid
    up_c = UserProfileSchema(
        name="Jane Doe",
        current_employment_status="employed",
        zip_code=None,
        what_is_the_user_looking_for=("Full-time admin roles in Montgomery County"),
    )
    if up_c.is_valid is not False:
        raise AssertionError("Expected invalid user profile when field missing")

    print("UserProfileSchema tests passed.")

    # Pretty print demo using utils
    from src.utils import format_as_json

    example_state = ReceptionistSubgraphState()
    print("\nPretty JSON for default ReceptionistSubgraphState:\n")
    print(format_as_json(example_state))

# %%
