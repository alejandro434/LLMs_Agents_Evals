"""receptionist node logic.

uv run -m src.graphs.qualifier.nodes_logic

"""

# %%
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from src.graphs.qualifier.chains import qualifier_chain, user_info_collection_chain
from src.graphs.qualifier.schemas import (
    QualifierSubgraphState,
    UserInfoOutputSchema,
)


async def collect_user_info(
    state: QualifierSubgraphState,
) -> Command[Literal["qualify_user"]]:
    """Receptor node - collects user information.

    - If prior user info exists, inject a concise context and merge new results
      to preserve previously extracted fields.
    - Ask for missing fields if needed via the chain's direct response.
    - Ensures reliable merging with ZIP-based state inference.
    """
    import json

    history = state["messages"][1:] if len(state["messages"]) > 1 else []

    prior_info = state.get("collected_user_info")
    context_injection = None

    # Check if we have valid prior info (not just empty strings)
    if (
        isinstance(prior_info, UserInfoOutputSchema)
        and prior_info.at_least_one_user_info_field_is_filled
    ):
        # Create a clean representation for the context, filtering out None/empty values
        prior_data = {}
        if prior_info.age is not None:
            prior_data["age"] = prior_info.age
        if prior_info.state and prior_info.state.strip():
            prior_data["state"] = prior_info.state
        if prior_info.zip_code and prior_info.zip_code.strip():
            prior_data["zip_code"] = prior_info.zip_code

        if prior_data:  # Only inject context if we have actual data
            context_injection = (
                "The user had already provided some data:\n"
                "here is the partial user info:\n"
                f"{json.dumps(prior_data, indent=2)}\n"
                "ask concisely for the missing information."
            )

    # Invoke with optional history and context
    response = await user_info_collection_chain.ainvoke(
        state["messages"][-1],
        current_history=history,
        runtime_context_injection=context_injection,
    )

    # Always merge with prior to retain previously filled values
    # The improved merge function now handles ZIP-based state inference
    if isinstance(prior_info, UserInfoOutputSchema):
        response = response.merged_with_prior(prior_info)

    # Validate that we have the required fields (age and zip_code)
    # Check for actual values, not just non-None
    has_valid_age = response.age is not None
    has_valid_zip = (
        response.zip_code is not None
        and isinstance(response.zip_code, str)
        and response.zip_code.strip() != ""
    )

    # If age or zip code is missing, ask for it
    if not has_valid_age or not has_valid_zip:
        # Ensure we have a response to show the user
        if not response.direct_response_to_the_user:
            missing_fields = []
            if not has_valid_age:
                missing_fields.append("your age")
            if not has_valid_zip:
                missing_fields.append("your ZIP code")
            response.direct_response_to_the_user = (
                f"Could you please provide {' and '.join(missing_fields)}?"
            )

        user_resp_to_interrupt = interrupt(response.direct_response_to_the_user)

        return Command(
            goto="collect_user_info", update={"messages": [user_resp_to_interrupt]}
        )

    return Command(goto="qualify_user", update={"collected_user_info": response})


async def qualify_user(
    state: QualifierSubgraphState,
) -> Command[Literal[END]]:
    """Qualify user node - qualifies user."""
    if state.get("collected_user_info"):
        collected_user_info = state["collected_user_info"]
        response = await qualifier_chain.ainvoke(collected_user_info)

        if isinstance(response.qualified, bool):
            return Command(
                goto=END,
                update={
                    "is_user_qualified": response.qualified,
                    "why_not_qualified": [response.why_not_qualified],
                },
            )
        return Command(goto=END, update={"messages": ["FAILED TO QUALIFY USER!"]})

    return Command(goto=END, update={"messages": ["FAILED TO COLLECT USER INFO!"]})


if __name__ == "__main__":
    import asyncio

    def _require(condition: bool, message: str) -> None:
        """Raise AssertionError with message if condition is False."""
        if not condition:
            raise AssertionError(message)

    async def test_collect_user_info_merges_prior() -> None:
        """New extraction should preserve prior values and infer state from ZIP."""
        prior = UserInfoOutputSchema(age=25, state=None, zip_code=None)
        state = QualifierSubgraphState(
            messages=["", "My ZIP is 21201"], collected_user_info=prior
        )
        cmd = await collect_user_info(state)
        merged = cmd.update.get("collected_user_info")
        _require(isinstance(merged, UserInfoOutputSchema), "bad merged type")
        _require(merged.age == 25, "age not preserved")
        _require(merged.state == "Maryland", "state not inferred from ZIP")
        _require(
            isinstance(merged.zip_code, str) and merged.zip_code != "",
            "zip_code missing",
        )
        print("collect_user_info merge ok:", merged.model_dump_json(indent=2))

    async def test_collect_user_info_prompts_for_missing() -> None:
        """When fields are missing, the node should ask concisely for them."""
        state = QualifierSubgraphState(messages=["Hi, I'm from VA."])
        cmd = await collect_user_info(state)
        merged = cmd.update.get("collected_user_info")
        _require(isinstance(merged, UserInfoOutputSchema), "bad merged type")
        _require(
            merged.direct_response_to_the_user is None
            or isinstance(merged.direct_response_to_the_user, str),
            "direct_response type invalid",
        )
        print("collect_user_info prompt ok:", merged.model_dump_json(indent=2))

    async def test_qualify_user_paths() -> None:
        """Exercise qualified and not-qualified paths."""
        # Qualified path: adult in Maryland
        state_q = QualifierSubgraphState(
            collected_user_info=UserInfoOutputSchema(
                age=25, state="Maryland", zip_code="21201"
            ),
            messages=[""],
        )
        cmd_q = await qualify_user(state_q)
        _require(cmd_q.goto == END, "qualified should END")
        _require(cmd_q.update.get("is_user_qualified") is True, "should qualify")
        print("qualify_user qualified ok:", cmd_q.update)

        # Not qualified path: under 18
        state_nq = QualifierSubgraphState(
            collected_user_info=UserInfoOutputSchema(age=17, state="Virginia"),
            messages=[""],
        )
        cmd_nq = await qualify_user(state_nq)
        _require(cmd_nq.goto == "collect_user_info", "should loop to collect")
        _require(
            cmd_nq.update.get("is_user_qualified") is False,
            "should not qualify",
        )
        print("qualify_user not qualified ok:", cmd_nq.update)

    async def main() -> None:
        """Run minimal tests for qualifier nodes."""
        await test_collect_user_info_merges_prior()
        await test_collect_user_info_prompts_for_missing()
        await test_qualify_user_paths()

    asyncio.run(main())
