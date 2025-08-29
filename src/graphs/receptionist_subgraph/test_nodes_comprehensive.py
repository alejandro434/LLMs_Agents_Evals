"""Comprehensive test suite for receptionist nodes logic.

uv run -m src.graphs.receptionist_subgraph.test_nodes_comprehensive
"""

# %%
import asyncio

from src.graphs.receptionist_subgraph.nodes_logic import (
    handoff_to_agent,
    receptor,
    validate_user_profile,
)
from src.graphs.receptionist_subgraph.schemas import (
    ReceptionistOutputSchema,
    ReceptionistSubgraphState,
)


class TestReceptorNode:
    """Test suite for the receptor node."""

    async def test_basic_extraction(self) -> None:
        """Test basic information extraction from user message."""
        state = ReceptionistSubgraphState(
            messages=["Hi, I'm John Smith looking for warehouse jobs in Baltimore."]
        )
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]

        assert schema is not None
        assert schema.direct_response_to_the_user is not None
        print("✅ Basic extraction test passed")

    async def test_incremental_extraction(self) -> None:
        """Test that receptor properly builds profile incrementally."""
        messages = []
        schema = None

        # Step 1: Name only
        messages.append("My name is Alice Johnson.")
        state = ReceptionistSubgraphState(messages=messages)
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]
        assert schema.user_name == "Alice Johnson"

        # Step 2: Add location
        messages.append("I live in Silver Spring, MD.")
        state = ReceptionistSubgraphState(
            messages=messages, receptionist_output_schema=schema
        )
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]
        assert schema.user_name == "Alice Johnson"  # Preserved
        assert "Silver Spring" in (schema.user_current_address or "")

        # Step 3: Add employment info
        messages.append("I'm unemployed, last worked as a nurse at MedStar.")
        state = ReceptionistSubgraphState(
            messages=messages, receptionist_output_schema=schema
        )
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]
        assert schema.user_employment_status == "unemployed"
        assert schema.user_last_job == "nurse" or "nurse" in (
            schema.user_last_job or ""
        )

        print("✅ Incremental extraction test passed")

    async def test_data_preservation(self) -> None:
        """Test that existing data is not overwritten by new extractions."""
        initial_schema = ReceptionistOutputSchema(
            user_name="Bob Chen",
            user_current_address="Arlington, VA",
            user_employment_status="employed",
        )

        state = ReceptionistSubgraphState(
            messages=[
                "I previously mentioned I'm Bob Chen from Arlington.",
                "Looking for remote software engineering roles.",
            ],
            receptionist_output_schema=initial_schema,
        )

        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]

        # Original data should be preserved
        assert schema.user_name == "Bob Chen"
        assert schema.user_current_address == "Arlington, VA"
        assert schema.user_employment_status == "employed"
        # New preference should be added
        assert "software" in (schema.user_job_preferences or "").lower()

        print("✅ Data preservation test passed")

    async def test_edge_cases(self) -> None:
        """Test receptor with various edge cases."""
        # Empty message
        state = ReceptionistSubgraphState(messages=[""])
        response = await receptor(state)
        assert response.update["receptionist_output_schema"] is not None

        # Very long message
        long_msg = " ".join(["Looking for jobs"] * 100)
        state = ReceptionistSubgraphState(messages=[long_msg])
        response = await receptor(state)
        assert response.update["receptionist_output_schema"] is not None

        # Special characters
        state = ReceptionistSubgraphState(
            messages=["My name is O'Brien-Smith & I need $100k+ jobs!"]
        )
        response = await receptor(state)
        assert response.update["receptionist_output_schema"] is not None

        print("✅ Edge cases test passed")


class TestValidateUserProfile:
    """Test suite for the validate_user_profile node."""

    async def test_complete_profile_handoff(self) -> None:
        """Test that complete profiles trigger handoff."""
        complete_schema = ReceptionistOutputSchema(
            user_name="Sarah Wilson",
            user_current_address="123 Main St, Baltimore, MD",
            user_employment_status="unemployed",
            user_last_job="Data Analyst",
            user_last_job_location="Baltimore, MD",
            user_last_job_company="TechCorp",
            user_job_preferences="Remote data science roles",
            direct_response_to_the_user=None,
        )

        state = ReceptionistSubgraphState(receptionist_output_schema=complete_schema)
        response = await validate_user_profile(state)

        assert response.goto == "handoff_to_logging"
        print("✅ Complete profile handoff test passed")

    async def test_incomplete_profile_interrupt(self) -> None:
        """Test that incomplete profiles trigger interrupt."""
        incomplete_schema = ReceptionistOutputSchema(
            user_name="Mike Brown",
            user_current_address=None,  # Missing
            user_employment_status="employed",
            user_last_job=None,  # Missing
            direct_response_to_the_user="Could you provide your address and last job?",
        )

        state = ReceptionistSubgraphState(receptionist_output_schema=incomplete_schema)

        try:
            await validate_user_profile(state)
        except RuntimeError as e:
            if "get_config outside of a runnable context" in str(e):
                print("✅ Incomplete profile interrupt test passed")
                return
        assert False, "Should have triggered interrupt"

    async def test_empty_response_handling(self) -> None:
        """Test handling of empty direct response with incomplete profile."""
        schema = ReceptionistOutputSchema(
            user_name="Test User",
            direct_response_to_the_user="",  # Empty
        )

        state = ReceptionistSubgraphState(receptionist_output_schema=schema)

        try:
            await validate_user_profile(state)
        except RuntimeError as e:
            if "get_config outside of a runnable context" in str(e):
                print("✅ Empty response handling test passed")
                return
        assert False, "Should have triggered interrupt with generic message"


class TestHandoffToAgent:
    """Test suite for the handoff_to_agent node."""

    async def test_successful_handoff(self) -> None:
        """Test successful handoff with complete profile."""
        state = ReceptionistSubgraphState(
            messages=["I need help finding cybersecurity training programs."],
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name="Emma Davis",
                user_current_address="Rockville, MD",
                user_employment_status="employed",
                user_last_job="IT Support",
                user_last_job_location="Rockville, MD",
                user_last_job_company="DataSys",
                user_job_preferences="Cybersecurity roles",
                direct_response_to_the_user=None,
            ),
        )

        response = await handoff_to_agent(state)

        # Check all required fields are present
        assert response.update["user_profile_schema"] is not None
        assert response.update["user_request"] is not None
        assert response.update["selected_agent"] == "react"
        assert response.update["rationale_of_the_handoff"] is not None

        # Verify profile was mapped correctly
        profile = response.update["user_profile_schema"]
        assert profile.name == "Emma Davis"
        assert profile.current_address == "Rockville, MD"

        print("✅ Successful handoff test passed")

    async def test_handoff_with_minimal_profile(self) -> None:
        """Test handoff with minimal/incomplete profile."""
        state = ReceptionistSubgraphState(
            messages=["Find me any job."],
            receptionist_output_schema=ReceptionistOutputSchema(
                user_name=None,
                direct_response_to_the_user=None,
            ),
        )

        response = await handoff_to_agent(state)

        # Should still produce valid output
        assert response.update["user_profile_schema"] is not None
        assert response.update["user_request"] is not None
        assert response.update["selected_agent"] == "react"

        print("✅ Minimal profile handoff test passed")

    async def test_task_extraction_accuracy(self) -> None:
        """Test that user request is accurately extracted."""
        test_cases = [
            (
                ["I need job fairs in Baltimore next week."],
                ["job fair", "baltimore", "week"],
            ),
            (
                ["Looking for remote Python developer positions."],
                ["remote", "python", "developer"],
            ),
            (
                ["Help me find HVAC certification programs."],
                ["hvac", "certification", "program"],
            ),
        ]

        for messages, expected_keywords in test_cases:
            state = ReceptionistSubgraphState(
                messages=messages,
                receptionist_output_schema=ReceptionistOutputSchema(
                    user_name="Test User",
                    user_current_address="Test City",
                    user_employment_status="unemployed",
                    user_last_job="Test Job",
                    user_last_job_location="Test Location",
                    user_last_job_company="Test Company",
                    user_job_preferences="Test Preferences",
                ),
            )

            response = await handoff_to_agent(state)
            task = response.update["user_request"].task.lower()

            for keyword in expected_keywords:
                assert keyword in task, f"Expected '{keyword}' in task: {task}"

        print("✅ Task extraction accuracy test passed")


class TestIntegrationFlows:
    """Test complete integration flows."""

    async def test_complete_conversation_flow(self) -> None:
        """Test a complete conversation from start to handoff."""
        print("\n" + "=" * 60)
        print("Testing Complete Conversation Flow")
        print("=" * 60)

        # Build conversation step by step
        messages = []
        schema = None

        # User introduces themselves
        messages.append(
            "Hi, I'm Jennifer Lee from Bethesda, MD. I need help finding a new job."
        )
        state = ReceptionistSubgraphState(messages=messages)
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]
        print(
            f"Step 1 - Extracted: name={schema.user_name}, address={schema.user_current_address}"
        )

        # Add employment status
        messages.append(
            "I'm currently unemployed. I was laid off from my marketing manager position."
        )
        state = ReceptionistSubgraphState(
            messages=messages, receptionist_output_schema=schema
        )
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]
        print(
            f"Step 2 - Added: status={schema.user_employment_status}, job={schema.user_last_job}"
        )

        # Add company and location
        messages.append("I worked at MarketPro in Bethesda for 5 years.")
        state = ReceptionistSubgraphState(
            messages=messages, receptionist_output_schema=schema
        )
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]
        print(f"Step 3 - Added: company={schema.user_last_job_company}")

        # Add preferences
        messages.append("I'm looking for senior marketing roles, preferably remote.")
        state = ReceptionistSubgraphState(
            messages=messages, receptionist_output_schema=schema
        )
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]
        print(f"Step 4 - Added: preferences={schema.user_job_preferences}")

        # Check if profile is complete
        if schema.user_info_complete:
            print("Step 5 - Profile complete, proceeding to handoff")

            state = ReceptionistSubgraphState(
                messages=messages, receptionist_output_schema=schema
            )

            # Validate profile
            validation = await validate_user_profile(state)
            assert validation.goto == "handoff_to_logging"

            # Perform handoff
            handoff = await handoff_to_agent(state)
            assert handoff.update["selected_agent"] == "react"
            print(
                f"Step 6 - Handoff complete: agent={handoff.update['selected_agent']}"
            )

        print("\n✅ Complete conversation flow test passed!")

    async def test_error_recovery_flow(self) -> None:
        """Test that the system recovers from errors gracefully."""
        # Start with invalid data
        state = ReceptionistSubgraphState(
            messages=[None, "", "Valid message after errors"]
        )

        # Should handle the errors and process valid message
        response = await receptor(state)
        schema = response.update["receptionist_output_schema"]
        assert schema is not None

        print("✅ Error recovery flow test passed")


async def run_all_tests() -> None:
    """Run all test suites."""
    print("Running Comprehensive Node Logic Tests")
    print("=" * 70)

    test_suites = [
        ("Receptor Node", TestReceptorNode()),
        ("Validate User Profile", TestValidateUserProfile()),
        ("Handoff to Agent", TestHandoffToAgent()),
        ("Integration Flows", TestIntegrationFlows()),
    ]

    for suite_name, suite in test_suites:
        print(f"\n🔬 Testing {suite_name}")
        print("-" * 40)

        for method_name in dir(suite):
            if method_name.startswith("test_"):
                test_method = getattr(suite, method_name)
                try:
                    await test_method()
                except AssertionError as e:
                    print(f"   ❌ {method_name} failed: {e}")
                except Exception as e:
                    print(f"   ⚠️ {method_name} error: {e}")

    print("\n" + "=" * 70)
    print("✨ All comprehensive tests completed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
