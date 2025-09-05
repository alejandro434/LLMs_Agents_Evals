"""Schemas for qualifier subgraph outputs.

These schemas are used by the qualifier subgraph chains. When a US ZIP code is
provided, downstream logic and prompts should infer the corresponding state and
populate the ``state`` field accordingly. In case of conflicts between a text
state and a ZIP-derived state, prefer the ZIP-derived state.
"""

# %%
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


def _require(condition: bool, message: str) -> None:
    """Raise AssertionError with message if condition is False.

    Used instead of bare asserts to satisfy linting rules.
    """
    if not condition:
        raise AssertionError(message)


class UserInfoOutputSchema(BaseModel):
    """User info output schema."""

    age: int | None = Field(
        default=None,
        description="The age of the user. Under 18 years old is not qualified.",
    )
    state: str | None = Field(
        default=None,
        description=(
            "The U.S. state where the user is located. If a ZIP code is"
            " provided, infer the state from the ZIP and prefer it over any"
            " conflicting textual state. ONLY states of Maryland and Virginia"
            " are qualified."
        ),
    )
    zip_code: str | None = Field(
        default=None,
        description=(
            "The US ZIP code of the user. If present, use it to infer the"
            " state. ONLY residents of Maryland and Virginia are qualified"
            " for the job."
        ),
    )
    direct_response_to_the_user: str | None = Field(
        default=None,
        description="The direct response to the user asking for either their age or zip code.",
    )

    @property
    def at_least_one_user_info_field_is_filled(self) -> bool:
        """Return True if any informational field (age/state/zip) is present and non-empty."""
        return (
            self.age is not None
            or (self.state is not None and str(self.state).strip() != "")
            or (self.zip_code is not None and str(self.zip_code).strip() != "")
        )

    def merged_with_prior(
        self, prior: "UserInfoOutputSchema"
    ) -> "UserInfoOutputSchema":
        """Merge with a prior instance, preferring current non-None/non-empty values.

        - For each field (age/state/zip_code), choose the new value if not None/empty;
          otherwise, keep the prior value.
        - If ZIP code is present after merge, re-infer state from ZIP.
        - Preserve the current direct_response_to_the_user.
        """
        if not isinstance(prior, UserInfoOutputSchema):
            return self

        merged: dict[str, object | None] = {}
        for field_name in ["age", "state", "zip_code"]:
            new_value = getattr(self, field_name, None)
            old_value = getattr(prior, field_name, None)

            # For string fields, treat empty strings as None
            if field_name in ["state", "zip_code"]:
                if isinstance(new_value, str) and new_value.strip() == "":
                    new_value = None
                if isinstance(old_value, str) and old_value.strip() == "":
                    old_value = None

            merged[field_name] = new_value if new_value is not None else old_value

        # Create the merged instance
        result = UserInfoOutputSchema(
            direct_response_to_the_user=self.direct_response_to_the_user,
            **merged,
        )

        # If we have a ZIP code after merging, ensure state is inferred from it
        if result.zip_code:
            from src.graphs.qualifier.chains import (
                _extract_zip5,
                _infer_state_from_zip,
            )

            zip5 = _extract_zip5(result.zip_code)
            if zip5:
                inferred_state = _infer_state_from_zip(zip5)
                if inferred_state:
                    result = result.model_copy(update={"state": inferred_state})

        return result


class QualifierOutputSchema(BaseModel):
    """Qualifier output schema."""

    qualified: bool | None = Field(
        default=None,
        description=(
            "Whether the user is qualified. Under 18 years old is not"
            " qualified. ONLY residents of Maryland (MD) and Virginia (VA)"
            " are qualified. If a ZIP is provided, infer the state from the"
            " ZIP and prefer it over any conflicting textual state."
        ),
    )

    why_not_qualified: str | None = Field(
        default=None, description="Explanation for the user why they are not qualified."
    )


class QualifierSubgraphState(MessagesState):
    """Qualifier subgraph state."""

    collected_user_info: UserInfoOutputSchema = Field(
        default_factory=UserInfoOutputSchema
    )
    user_zip_and_age: UserInfoOutputSchema = Field(default_factory=UserInfoOutputSchema)
    is_user_qualified: QualifierOutputSchema = Field(
        default_factory=QualifierOutputSchema
    )
    why_not_qualified: str | None = Field(default=None)


if __name__ == "__main__":
    # Simple demonstration / test
    example_user_info = UserInfoOutputSchema(
        age=25,
        state="Maryland",
        zip_code="21201",
    )
    _require(example_user_info.age == 25, "age mismatch")
    _require(example_user_info.state == "Maryland", "state mismatch")
    _require(example_user_info.zip_code == "21201", "zip mismatch")
    print("UserInfoOutputSchema OK:", example_user_info.model_dump())

    example_qualifier = QualifierOutputSchema(
        qualified=True,
        why_not_qualified=None,
    )
    _require(example_qualifier.qualified is True, "qualified mismatch")
    _require(
        example_qualifier.why_not_qualified is None,
        "why_not_qualified should be None",
    )
    print("QualifierOutputSchema OK:", example_qualifier.model_dump())
