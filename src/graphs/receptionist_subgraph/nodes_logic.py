"""receptionist node logic.

uv run -m src.graphs.receptionist_subgraph.nodes_logic

"""

# %%
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from src.graphs.receptionist_subgraph.chains import (
    profiling_chain,
    receptionist_chain,
)
from src.graphs.receptionist_subgraph.schemas import (
    ReceptionistOutputSchema,
    ReceptionistSubgraphState,
)
from src.utils import format_command


async def receptor(
    state: ReceptionistSubgraphState,
) -> Command[Literal["validate_user_profile"]]:
    """Receptionist node."""
    history = state["messages"][1:] if len(state["messages"]) > 1 else None
    prior_schema = state.get("receptionist_output_schema")
    context_injection = None
    if (
        isinstance(prior_schema, ReceptionistOutputSchema)
        and prior_schema.at_least_one_user_profile_field_is_filled
    ):
        context_injection = (
            "The user had already provided some data:\n"
            "here is the partial user profile:\n"
            f"{prior_schema.model_dump_json(indent=2)}\n"
            "ask for the missing information."
        )

    response = await receptionist_chain.ainvoke(
        state["messages"][-1],
        current_history=history,
        runtime_context_injection=context_injection,
    )
    return Command(
        goto="validate_user_profile",
        update={"receptionist_output_schema": response},
    )


async def validate_user_profile(
    state: ReceptionistSubgraphState,
) -> Command[Literal["handoff_to_logging", "receptor"]]:
    """User profile node."""
    if state.get("receptionist_output_schema").user_info_complete:
        return Command(goto="handoff_to_logging")

    user_answer = interrupt(
        state.get("receptionist_output_schema").direct_response_to_the_user
    )
    return Command(goto="receptor", update={"messages": [user_answer]})


async def handoff_to_logging(
    state: ReceptionistSubgraphState,
) -> Command[Literal[END]]:
    """Handoff to logging node."""
    user_profile = await profiling_chain.ainvoke(
        state.get("receptionist_output_schema")
    )
    return Command(goto=END, update={"user_profile_schema": user_profile})


if __name__ == "__main__":
    import asyncio
    import os

    async def test_receptor() -> None:
        """Test receptor node."""
        state = ReceptionistSubgraphState(
            messages=["Hi, I'm looking for an entry-level retail job."]
        )
        response = await receptor(state)
        print(format_command(response))

    asyncio.run(test_receptor())

    async def test_receptor_with_history_and_context() -> None:
        """Test receptor with history and partial profile for context injection."""
        state = ReceptionistSubgraphState(
            messages=[
                "Hi, I'm looking for an entry-level retail job.",
                "Earlier I shared my name and neighborhood.",
            ],
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name="John Doe",
                direct_response_to_the_user=None,
            ),
        )
        response = await receptor(state)
        print(format_command(response))

    if os.environ.get("RUN_ADVANCED_RECEPTOR_TEST") == "1":
        asyncio.run(test_receptor_with_history_and_context())

    async def test_receptor_with_structured_history() -> None:
        """Test receptor with dict-based history entries."""
        state = ReceptionistSubgraphState(
            messages=[
                "I'm interested in retail jobs.",
                {"human": "Previously, I mentioned I'm in Arlington."},
                {"ai": "What is your name and current address?"},
                "Can you help me find entry-level roles?",
            ],
        )
        response = await receptor(state)
        print(format_command(response))

    if os.environ.get("RUN_ADVANCED_RECEPTOR_TEST") == "1":
        asyncio.run(test_receptor_with_structured_history())

    async def test_validate_user_profile() -> None:
        """Test validate_user_profile node."""
        state = ReceptionistSubgraphState(
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name="John Doe",
                user_address="123 Main St, Anytown, USA",
                user_employment_status="employed",
                user_last_job="Sales Associate",
                user_last_job_location="Anytown, USA",
                user_last_job_company="RetailCo",
                user_job_preferences=("Entry-level retail roles in Anytown, full-time"),
                direct_response_to_the_user=(
                    "Sure, I'll need to know your name and current address."
                ),
            ),
        )
        response = await validate_user_profile(state)
        print(format_command(response))

    asyncio.run(test_validate_user_profile())

    async def test_handoff_to_logging() -> None:
        """Test handoff_to_logging node."""
        state = ReceptionistSubgraphState(
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name="John Doe",
                user_address="123 Main St, Anytown, USA",
                user_employment_status="employed",
                user_last_job="Sales Associate",
                user_last_job_location="Anytown, USA",
                user_last_job_company="RetailCo",
                user_job_preferences=("Entry-level retail roles in Anytown, full-time"),
                direct_response_to_the_user=None,
            ),
        )
        response = await handoff_to_logging(state)
        print(format_command(response))

    asyncio.run(test_handoff_to_logging())

    async def test_handoff_to_logging_minimal() -> None:
        """Test handoff_to_logging with minimal valid profile signal."""
        state = ReceptionistSubgraphState(
            receptionist_output_schema=ReceptionistOutputSchema(
                direct_response_to_the_user=None,
                user_name=None,
            ),
        )
        response = await handoff_to_logging(state)
        print(format_command(response))

    if os.environ.get("RUN_ADVANCED_RECEPTOR_TEST") == "1":
        asyncio.run(test_handoff_to_logging_minimal())

    async def test_validate_user_profile_complete_profile_case1() -> None:
        """validate_user_profile should handoff when profile is complete."""
        complete = ReceptionistOutputSchema(
            direct_response_to_the_user=None,
            user_name="Alice Smith",
            user_current_address="789 Pine St, Fairfax, VA",
            user_employment_status="employed",
            user_last_job="Cashier",
            user_last_job_location="Fairfax, VA",
            user_last_job_company="QuickMart",
            user_job_preferences="Full-time retail roles in Fairfax",
        )
        state = ReceptionistSubgraphState(receptionist_output_schema=complete)
        response = await validate_user_profile(state)
        if getattr(response, "goto", None) != "handoff_to_logging":
            raise AssertionError(
                f"Expected handoff_to_logging, got {getattr(response, 'goto', None)}"
            )
        print(format_command(response))

    asyncio.run(test_validate_user_profile_complete_profile_case1())

    async def test_validate_user_profile_complete_profile_case2() -> None:
        """Another complete profile path to handoff."""
        complete = ReceptionistOutputSchema(
            direct_response_to_the_user="Thanks! I have all I need.",
            user_name="Bob Jones",
            user_current_address="1600 Main St, Rockville, MD",
            user_employment_status="self-employed",
            user_last_job="Owner",
            user_last_job_location="Rockville, MD",
            user_last_job_company="Bob's Bikes",
            user_job_preferences="Operations management, 40 hrs, hybrid",
        )
        state = ReceptionistSubgraphState(receptionist_output_schema=complete)
        response = await validate_user_profile(state)
        if getattr(response, "goto", None) != "handoff_to_logging":
            raise AssertionError(
                f"Expected handoff_to_logging, got {getattr(response, 'goto', None)}"
            )
        print(format_command(response))

    asyncio.run(test_validate_user_profile_complete_profile_case2())

# %%
