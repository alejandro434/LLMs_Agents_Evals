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
    """
    history = state["messages"][1:] if len(state["messages"]) > 1 else None

    prior_info = state.get("collected_user_info")
    context_injection = None
    if (
        isinstance(prior_info, UserInfoOutputSchema)
        and prior_info.at_least_one_user_info_field_is_filled
    ):
        context_injection = (
            "The user had already provided some data:\n"
            "here is the partial user info:\n"
            f"{prior_info.model_dump_json(indent=2)}\n"
            "ask concisely for the missing information."
        )

    # Invoke with optional history and context
    response = await user_info_collection_chain.ainvoke(
        state["messages"][-1],
        current_history=history,
        runtime_context_injection=context_injection,
    )

    # Merge with prior to retain previously filled values
    if isinstance(prior_info, UserInfoOutputSchema):
        response = response.merged_with_prior(prior_info)

    direct_response_to_the_user = response.direct_response_to_the_user
    update_payload: dict[str, object] = {"collected_user_info": response}
    # Avoid pushing None into the messages channel (MessageLike must be str/BaseMessage)
    if (
        isinstance(direct_response_to_the_user, str)
        and direct_response_to_the_user.strip() != ""
    ):
        update_payload["messages"] = [direct_response_to_the_user]

    # If age or zip code is missing, ask for it
    if response.age is None or response.zip_code is None:
        update_payload["messages"] = [response.direct_response_to_the_user]
        user_resp_to_interrupt = interrupt(response.direct_response_to_the_user)

        return Command(
            goto="collect_user_info", update={"messages": [user_resp_to_interrupt]}
        )

    return Command(goto="qualify_user", update=update_payload)


async def qualify_user(
    state: QualifierSubgraphState,
) -> Command[Literal[END]]:
    """Qualify user node - qualifies user."""
    collected_user_info = state["collected_user_info"]
    response = await qualifier_chain.ainvoke(collected_user_info)
    if response.qualified:
        return Command(goto=END, update={"is_user_qualified": response.qualified})

    update_payload: dict[str, object] = {"is_user_qualified": response.qualified}
    # Guard against None messages to satisfy LangChain MessageLike rules
    if (
        isinstance(response.why_not_qualified, str)
        and response.why_not_qualified.strip() != ""
    ):
        update_payload["messages"] = [response.why_not_qualified]
    return Command(goto=END, update=update_payload)


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
