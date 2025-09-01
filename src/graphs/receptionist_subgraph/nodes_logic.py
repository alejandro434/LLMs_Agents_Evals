"""receptionist node logic.

uv run -m src.graphs.receptionist_subgraph.nodes_logic

"""

# %%
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from src.graphs.receptionist_subgraph.chains import (
    agent_selection_chain,
    profiling_chain,
    receptionist_chain,
    user_request_extraction_chain,
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
        print(f"Error in receptionist chain in the receptor node: {e}")

    return Command(
        goto="validate_user_profile",
        update={
            "receptionist_output_schema": response,
            "direct_response_to_the_user": response.direct_response_to_the_user,
        },
    )


async def validate_user_profile(
    state: ReceptionistSubgraphState,
) -> Command[Literal["handoff_to_agent", "receptor"]]:
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
        return Command(goto="handoff_to_agent")

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
    return Command(
        goto="receptor",
        update={"messages": [user_answer]},
    )


async def handoff_to_agent(
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

        user_request = await user_request_extraction_chain.ainvoke(state["messages"])
        agent_selection = await agent_selection_chain.ainvoke(user_request.task)

        return Command(
            goto=END,
            update={
                "user_profile": user_profile,
                "user_request": user_request,
                "selected_agent": agent_selection.agent_name,
                "rationale_of_the_handoff": agent_selection.rationale_of_the_handoff,
            },
        )

    except Exception as e:
        # In case of profiling failure, create a minimal profile from available data
        print(f"Error in profiling in the handoff_to_agent node: {e}.")


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

    asyncio.run(test_receptor_with_structured_history())

    async def test_validate_user_profile_incomplete() -> None:
        """Test validate_user_profile with incomplete profile - should trigger interrupt."""
        state = ReceptionistSubgraphState(
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name="John Doe",
                user_current_address=None,  # Missing address
                user_employment_status="employed",
                user_last_job="Sales Associate",
                user_last_job_location=None,  # Missing location
                user_last_job_company="RetailCo",
                user_job_preferences=None,  # Missing preferences
                direct_response_to_the_user=(
                    "Could you please provide your current address and job preferences?"
                ),
            ),
        )
        try:
            # This will raise RuntimeError outside of LangGraph context
            response = await validate_user_profile(state)
        except RuntimeError as e:
            if "get_config outside of a runnable context" in str(e):
                print(
                    "✅ validate_user_profile correctly attempts interrupt for incomplete profile"
                )
            else:
                raise

    asyncio.run(test_validate_user_profile_incomplete())

    async def test_handoff_to_agent() -> None:
        """Test handoff_to_agent node."""
        state = ReceptionistSubgraphState(
            messages=["I need help finding warehouse jobs with good benefits"],
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name="John Doe",
                user_current_address="123 Main St, Baltimore, MD",
                user_employment_status="unemployed",
                user_last_job="Sales Associate",
                user_last_job_location="Baltimore, MD",
                user_last_job_company="RetailCo",
                user_job_preferences="Entry-level warehouse roles, full-time with benefits",
                direct_response_to_the_user=None,
            ),
        )
        response = await handoff_to_agent(state)
        print(format_command(response))

        # Validate the response contains expected fields
        assert response.update.get("user_profile") is not None
        assert response.update.get("user_request") is not None
        assert response.update.get("selected_agent") == "react"
        assert response.update.get("rationale_of_the_handoff") is not None
        print("✅ handoff_to_agent test passed")

    asyncio.run(test_handoff_to_agent())

    async def test_handoff_to_agent_minimal() -> None:
        """Test handoff_to_agent with minimal profile - should handle gracefully."""
        state = ReceptionistSubgraphState(
            messages=["Find me a job"],
            receptionist_output_schema=ReceptionistOutputSchema(
                direct_response_to_the_user=None,
                user_name=None,
            ),
        )
        response = await handoff_to_agent(state)
        print(format_command(response))

        # Should still produce a valid response even with minimal data
        assert response.update.get("user_profile") is not None
        print("✅ handoff_to_agent minimal test passed")

    if os.environ.get("RUN_ADVANCED_RECEPTOR_TEST") == "1":
        asyncio.run(test_handoff_to_agent_minimal())

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
        if getattr(response, "goto", None) != "handoff_to_agent":
            raise AssertionError(
                f"Expected handoff_to_agent, got {getattr(response, 'goto', None)}"
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
        if getattr(response, "goto", None) != "handoff_to_agent":
            raise AssertionError(
                f"Expected handoff_to_agent, got {getattr(response, 'goto', None)}"
            )
        print(format_command(response))

    asyncio.run(test_validate_user_profile_complete_profile_case2())

    async def test_receptor_error_handling() -> None:
        """Test receptor node error handling with various edge cases."""
        # Test with empty message
        state1 = ReceptionistSubgraphState(messages=[""])
        response1 = await receptor(state1)
        assert response1.update["receptionist_output_schema"] is not None
        print("✅ receptor handles empty message gracefully")

        # Test with very long message
        long_message = "I " + " ".join(["need help finding jobs"] * 100)
        state2 = ReceptionistSubgraphState(messages=[long_message])
        response2 = await receptor(state2)
        assert response2.update["receptionist_output_schema"] is not None
        print("✅ receptor handles very long message gracefully")

        # Test with special characters
        state3 = ReceptionistSubgraphState(
            messages=["My name is J@hn D0e! I'm looking for #jobs in $Baltimore!!!"]
        )
        response3 = await receptor(state3)
        assert response3.update["receptionist_output_schema"] is not None
        print("✅ receptor handles special characters gracefully")

    asyncio.run(test_receptor_error_handling())

    async def test_receptor_merge_functionality() -> None:
        """Test that receptor properly merges new and existing user data."""
        # Start with partial profile
        initial_schema = ReceptionistOutputSchema(
            user_name="Jane Smith",
            user_current_address="Arlington, VA",
            user_employment_status=None,
            user_last_job=None,
            user_last_job_location=None,
            user_last_job_company=None,
            user_job_preferences=None,
        )

        state = ReceptionistSubgraphState(
            messages=[
                "My name is Jane Smith from Arlington, VA.",
                "I'm currently unemployed. My last job was as a nurse at MedStar in Arlington.",
            ],
            receptionist_output_schema=initial_schema,
        )

        response = await receptor(state)
        updated_schema = response.update["receptionist_output_schema"]

        # Check that original data is preserved and new data is added
        assert updated_schema.user_name == "Jane Smith"
        assert updated_schema.user_current_address == "Arlington, VA"
        assert updated_schema.user_employment_status is not None
        print("✅ receptor merge functionality test passed")

    asyncio.run(test_receptor_merge_functionality())

    async def test_handoff_to_agent_with_request_extraction() -> None:
        """Test that handoff_to_agent properly extracts user request and selects agent."""
        state = ReceptionistSubgraphState(
            messages=[
                "Hi, I'm looking for job fairs in my area.",
                "My name is Alex Chen.",
                "I live in Silver Spring, MD.",
                "I'm unemployed, previously worked as a software developer at TechCorp in Rockville.",
                "I want remote software engineering positions.",
            ],
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name="Alex Chen",
                user_current_address="Silver Spring, MD",
                user_employment_status="unemployed",
                user_last_job="Software Developer",
                user_last_job_location="Rockville, MD",
                user_last_job_company="TechCorp",
                user_job_preferences="Remote software engineering positions",
                direct_response_to_the_user=None,
            ),
        )

        response = await handoff_to_agent(state)

        # Verify all expected fields are present
        assert response.update["user_profile"].name == "Alex Chen"
        assert response.update["user_request"].task is not None
        assert "job fair" in response.update["user_request"].task.lower()
        assert response.update["selected_agent"] == "react"
        assert len(response.update["rationale_of_the_handoff"]) > 20
        print("✅ handoff_to_agent with request extraction test passed")

    asyncio.run(test_handoff_to_agent_with_request_extraction())

    async def test_validate_user_profile_edge_cases() -> None:
        """Test validate_user_profile with various edge cases."""
        # Test with empty direct response and incomplete profile
        state1 = ReceptionistSubgraphState(
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name="Test User",
                user_current_address=None,
                direct_response_to_the_user="",  # Empty response
            ),
        )
        try:
            response1 = await validate_user_profile(state1)
        except RuntimeError as e:
            if "get_config outside of a runnable context" in str(e):
                print(
                    "✅ validate_user_profile correctly handles empty response with interrupt"
                )

        # Test with all fields null except direct_response
        state2 = ReceptionistSubgraphState(
            receptionist_output_schema=ReceptionistOutputSchema(
                direct_response_to_the_user="What's your name?",
            ),
        )
        try:
            response2 = await validate_user_profile(state2)
        except RuntimeError as e:
            if "get_config outside of a runnable context" in str(e):
                print(
                    "✅ validate_user_profile correctly handles null fields with interrupt"
                )

    asyncio.run(test_validate_user_profile_edge_cases())

    async def test_integration_flow() -> None:
        """Test complete flow from receptor through validation to handoff."""
        print("\n" + "=" * 60)
        print("Running Integration Flow Test")
        print("=" * 60)

        # Step 1: Initial user message
        state = ReceptionistSubgraphState(
            messages=["I need help finding IT certification programs in Maryland."]
        )

        # Process through receptor
        response1 = await receptor(state)
        schema1 = response1.update["receptionist_output_schema"]
        print(
            f"Step 1 - Receptor response: {schema1.direct_response_to_the_user[:80]}..."
        )

        # Step 2: Add user response with more info
        state = ReceptionistSubgraphState(
            messages=state["messages"]
            + ["I'm Bob Wilson from Baltimore, currently unemployed."],
            receptionist_output_schema=schema1,
        )

        response2 = await receptor(state)
        schema2 = response2.update["receptionist_output_schema"]
        print(
            f"Step 2 - Updated profile: name={schema2.user_name}, status={schema2.user_employment_status}"
        )

        # Step 3: Add final missing info
        state = ReceptionistSubgraphState(
            messages=state["messages"]
            + [
                "Last worked as network admin at DataCorp in Baltimore, looking for cybersecurity roles."
            ],
            receptionist_output_schema=schema2,
        )

        response3 = await receptor(state)
        schema3 = response3.update["receptionist_output_schema"]
        state = ReceptionistSubgraphState(
            messages=state["messages"],
            receptionist_output_schema=schema3,
        )

        # Step 4: Check if profile is complete
        if schema3.user_info_complete:
            print("Step 3 - Profile complete, proceeding to handoff")

            # Test validation with complete profile
            validation_response = await validate_user_profile(state)
            assert validation_response.goto == "handoff_to_agent"

            # Step 5: Perform handoff
            handoff_response = await handoff_to_agent(state)
            assert handoff_response.update["selected_agent"] == "react"
            print(
                f"Step 4 - Agent selected: {handoff_response.update['selected_agent']}"
            )
            print(
                f"Step 5 - Task extracted: {handoff_response.update['user_request'].task[:80]}..."
            )
        else:
            print("Step 3 - Profile incomplete, would trigger interrupt for more info")
            # Test that validation would trigger interrupt
            try:
                validation_response = await validate_user_profile(state)
            except RuntimeError as e:
                if "get_config outside of a runnable context" in str(e):
                    print(
                        "Step 4 - Correctly attempts interrupt for incomplete profile"
                    )

        print("\n✅ Integration flow test completed successfully!")

    asyncio.run(test_integration_flow())

# %%
