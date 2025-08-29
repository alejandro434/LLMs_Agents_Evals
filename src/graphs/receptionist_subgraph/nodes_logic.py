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
    """Receptionist node - extracts user information from conversation.

    This node processes user messages to extract profile information and
    determines what additional information is needed.

    Args:
        state: Current subgraph state containing messages and prior extractions

    Returns:
        Command to validate the extracted profile
    """
    # Get conversation history (excluding the current message)
    history = state["messages"][1:] if len(state["messages"]) > 1 else None

    # Check if we have prior extracted information to provide as context
    prior_schema = state.get("receptionist_output_schema")
    context_injection = None

    if (
        isinstance(prior_schema, ReceptionistOutputSchema)
        and prior_schema.at_least_one_user_profile_field_is_filled
    ):
        # Build context about previously extracted information
        context_injection = (
            "The user had already provided some data:\n"
            "here is the partial user profile:\n"
            f"{prior_schema.model_dump_json(indent=2)}\n"
            "ask for the missing information."
        )

    try:
        # Invoke the receptionist chain to extract/update user information
        response = await receptionist_chain.ainvoke(
            state["messages"][-1],
            current_history=history,
            runtime_context_injection=context_injection,
        )

        # CRITICAL: Merge new extraction with existing data to preserve all fields
        if prior_schema and isinstance(prior_schema, ReceptionistOutputSchema):
            # For each field, use the new value if provided, otherwise keep the old value
            merged_data = {}
            for field_name in [
                "user_name",
                "user_current_address",
                "user_employment_status",
                "user_last_job",
                "user_last_job_location",
                "user_last_job_company",
                "user_job_preferences",
            ]:
                new_value = getattr(response, field_name, None)
                old_value = getattr(prior_schema, field_name, None)
                # Use new value if it's not None, otherwise keep old value
                merged_data[field_name] = (
                    new_value if new_value is not None else old_value
                )

            # Create merged response with the direct response from the new extraction
            response = ReceptionistOutputSchema(
                direct_response_to_the_user=response.direct_response_to_the_user,
                **merged_data,
            )

    except Exception as e:
        # Handle chain invocation errors gracefully
        print(f"Error in receptionist chain: {e}")

        # Create a response asking for clarification
        response = ReceptionistOutputSchema(
            direct_response_to_the_user=(
                "I'm having trouble understanding. Could you please rephrase "
                "or provide more details about your employment situation and job preferences?"
            ),
            # Preserve any previously extracted fields
            user_name=prior_schema.user_name if prior_schema else None,
            user_current_address=prior_schema.user_current_address
            if prior_schema
            else None,
            user_employment_status=prior_schema.user_employment_status
            if prior_schema
            else None,
            user_last_job=prior_schema.user_last_job if prior_schema else None,
            user_last_job_location=prior_schema.user_last_job_location
            if prior_schema
            else None,
            user_last_job_company=prior_schema.user_last_job_company
            if prior_schema
            else None,
            user_job_preferences=prior_schema.user_job_preferences
            if prior_schema
            else None,
        )

    return Command(
        goto="validate_user_profile",
        update={"receptionist_output_schema": response},
    )


async def validate_user_profile(
    state: ReceptionistSubgraphState,
) -> Command[Literal["handoff_to_logging", "receptor"]]:
    """Validate user profile completeness and decide next action.

    This node checks if all required user information has been collected.
    If complete, it proceeds to handoff. If not, it interrupts to ask for
    missing information.

    Args:
        state: Current subgraph state

    Returns:
        Command to either handoff (if complete) or continue gathering info
    """
    receptionist_output = state.get("receptionist_output_schema")

    # Check if all user info is complete
    if receptionist_output.user_info_complete:
        # All required fields are present, proceed to profiling/handoff
        return Command(goto="handoff_to_logging")

    # Info is incomplete, interrupt to get user's response
    # The direct_response_to_the_user should contain the question for missing info
    response_to_user = receptionist_output.direct_response_to_the_user

    if not response_to_user:
        # Safety check: if no response but info incomplete, create a generic prompt
        response_to_user = (
            "I need a bit more information to help you better. "
            "Could you please provide any missing details about your name, address, "
            "employment status, or job preferences?"
        )

    user_answer = interrupt(response_to_user)
    return Command(goto="receptor", update={"messages": [user_answer]})


async def handoff_to_logging(
    state: ReceptionistSubgraphState,
) -> Command[Literal[END]]:
    """Handoff to logging node.

    This node performs the final profiling step, mapping the receptionist's
    extracted data to a standardized UserProfileSchema for downstream processing.

    Args:
        state: Current subgraph state containing receptionist_output_schema

    Returns:
        Command to end the subgraph with the mapped user profile
    """
    try:
        # Get the receptionist output for profiling
        receptionist_output = state.get("receptionist_output_schema")

        # Convert to UserProfileSchema using the profiling chain
        user_profile = await profiling_chain.ainvoke(receptionist_output)

        # Validate the profile was successfully mapped
        if not user_profile.is_valid:
            # Log warning but still proceed - downstream agents can handle incomplete data
            print("Warning: User profile validation failed. Missing fields in profile.")

        return Command(goto=END, update={"user_profile_schema": user_profile})

    except Exception as e:
        # In case of profiling failure, create a minimal profile from available data
        print(f"Error in profiling: {e}. Creating fallback profile.")

        # Create a fallback UserProfileSchema with available data
        from src.graphs.receptionist_subgraph.schemas import UserProfileSchema

        fallback_profile = UserProfileSchema(
            name=receptionist_output.user_name,
            current_address=receptionist_output.user_current_address,
            employment_status=receptionist_output.user_employment_status,
            last_job=receptionist_output.user_last_job,
            last_job_location=receptionist_output.user_last_job_location,
            last_job_company=receptionist_output.user_last_job_company,
            job_preferences=receptionist_output.user_job_preferences,
        )

        return Command(goto=END, update={"user_profile_schema": fallback_profile})


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
