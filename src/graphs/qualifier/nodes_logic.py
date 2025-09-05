"""Qualifier subgraph node logic.

This module contains the node logic for the qualifier subgraph that collects
user information (age, state, ZIP) and determines job qualification based on:
- Age: Must be 18 or older
- Location: Must be in Maryland, Virginia, or Washington D.C.

uv run -m src.graphs.qualifier.nodes_logic
"""

# %%
import json
import logging
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, interrupt

from src.graphs.qualifier.chains import (
    qualifier_chain,
    user_info_collection_chain,
)
from src.graphs.qualifier.schemas import (
    QualifierOutputSchema,
    QualifierSubgraphState,
    UserInfoOutputSchema,
)


# Configure logging
logger = logging.getLogger(__name__)


async def collect_user_info(
    state: QualifierSubgraphState,
) -> Command[Literal["qualify_user", "collect_user_info"]]:
    """Collect user information node - gathers age, state, and ZIP code.

    This node:
    1. Extracts user info from the latest message
    2. Merges with any prior collected information
    3. Validates that required fields (age and ZIP) are present
    4. Interrupts to ask for missing information if needed
    5. Proceeds to qualification once all info is collected

    Args:
        state: Current qualifier subgraph state

    Returns:
        Command to either:
        - Loop back to collect_user_info (if info missing)
        - Proceed to qualify_user (if info complete)

    Raises:
        ValueError: If state messages are empty or invalid
    """
    # Validate state
    if not state.get("messages"):
        logger.error("No messages in state")
        raise ValueError("State must contain at least one message")

    # Extract conversation history (excluding the first system message if present)
    history = state["messages"][1:] if len(state["messages"]) > 1 else []

    # Get any previously collected info
    prior_info = state.get("collected_user_info")
    context_injection = None

    # If we have valid prior info, prepare context for the chain
    if (
        isinstance(prior_info, UserInfoOutputSchema)
        and prior_info.at_least_one_user_info_field_is_filled
    ):
        # Build clean context with only non-empty fields
        prior_data = {}
        if prior_info.age is not None:
            prior_data["age"] = prior_info.age
        if prior_info.state and prior_info.state.strip():
            prior_data["state"] = prior_info.state
        if prior_info.zip_code and prior_info.zip_code.strip():
            prior_data["zip_code"] = prior_info.zip_code

        if prior_data:
            context_injection = (
                "The user had already provided some data:\n"
                "here is the partial user info:\n"
                f"{json.dumps(prior_data, indent=2)}\n"
                "ask concisely for the missing information."
            )
            logger.debug(f"Using prior data context: {prior_data}")

    # Invoke the user info collection chain
    try:
        response = await user_info_collection_chain.ainvoke(
            state["messages"][-1],
            current_history=history,
            runtime_context_injection=context_injection,
        )
    except Exception as e:
        logger.error(f"Error invoking user_info_collection_chain: {e}")
        # Return a safe fallback that asks for all info
        response = UserInfoOutputSchema(
            direct_response_to_the_user="Could you please provide your age and ZIP code?"
        )

    # Merge with prior info if available
    if prior_info and isinstance(prior_info, UserInfoOutputSchema):
        response = response.merged_with_prior(prior_info)
        logger.debug(f"Merged response: {response.model_dump()}")

    # Validate collected information
    has_valid_age = response.age is not None and response.age >= 0
    has_valid_zip = (
        response.zip_code is not None
        and isinstance(response.zip_code, str)
        and response.zip_code.strip() != ""
    )

    # Check if we need more information
    if not has_valid_age or not has_valid_zip:
        # Build a friendly request for missing fields
        missing_fields = []
        if not has_valid_age:
            missing_fields.append("your age")
        if not has_valid_zip:
            missing_fields.append("your ZIP code")

        # Ensure we have a response for the user
        if not response.direct_response_to_the_user:
            response.direct_response_to_the_user = (
                f"Could you please provide {' and '.join(missing_fields)}?"
            )

        logger.info(f"Missing fields: {missing_fields}")

        # Create interrupt message - this will pause execution in production
        # but we handle it gracefully in tests
        try:
            user_resp_to_interrupt = interrupt(response.direct_response_to_the_user)
            messages_update = [user_resp_to_interrupt]
        except RuntimeError as e:
            # In test environment without LangGraph runtime context
            logger.debug(f"Running outside LangGraph context (likely in tests): {e}")
            messages_update = [response.direct_response_to_the_user]

        # Loop back to collect more info, preserving what we have
        return Command(
            goto="collect_user_info",
            update={
                "messages": messages_update,
                "collected_user_info": response,  # Preserve partial info
            },
        )

    # All required info collected, proceed to qualification
    logger.info(f"User info complete: age={response.age}, zip={response.zip_code}")
    return Command(goto="qualify_user", update={"collected_user_info": response})


async def qualify_user(
    state: QualifierSubgraphState,
) -> Command[Literal[END]]:
    """Qualify user node - determines if user meets job requirements.

    Qualification criteria:
    - Age: Must be 18 or older
    - Location: Must be in MD, VA, or DC (determined by ZIP code)

    Args:
        state: Current qualifier subgraph state with collected_user_info

    Returns:
        Command to END with qualification result and reason if not qualified

    Note:
        Always proceeds to END regardless of qualification result.
        The parent graph handles what to do based on qualification status.
    """
    # Validate we have collected info
    collected_info = state.get("collected_user_info")
    if not collected_info or not isinstance(collected_info, UserInfoOutputSchema):
        logger.error("No collected user info available for qualification")
        return Command(
            goto=END,
            update={
                "is_user_qualified": False,
                "why_not_qualified": "Unable to collect required information",
            },
        )

    # Invoke the qualifier chain - construct a text input from collected info
    try:
        # Build a natural language input for the qualifier chain
        input_text = f"I'm {collected_info.age} years old"
        if collected_info.state:
            input_text += f" in {collected_info.state}"
        if collected_info.zip_code:
            input_text += f", ZIP {collected_info.zip_code}"

        response = await qualifier_chain.ainvoke(input_text)
    except Exception as e:
        logger.error(f"Error invoking qualifier_chain: {e}")
        return Command(
            goto=END,
            update={
                "is_user_qualified": False,
                "why_not_qualified": "Error occurred during qualification assessment",
            },
        )

    # Validate response
    if not isinstance(response, QualifierOutputSchema):
        logger.error(f"Invalid response type from qualifier_chain: {type(response)}")
        return Command(
            goto=END,
            update={
                "is_user_qualified": False,
                "why_not_qualified": "System error during qualification",
            },
        )

    # Extract qualification result
    is_qualified = response.qualified if isinstance(response.qualified, bool) else False
    why_not = response.why_not_qualified

    # Log the qualification decision
    if is_qualified:
        logger.info(
            f"User QUALIFIED: age={collected_info.age}, state={collected_info.state}"
        )
    else:
        logger.info(f"User NOT qualified: {why_not}")

    # Return the qualification result
    return Command(
        goto=END,
        update={
            "is_user_qualified": is_qualified,
            "why_not_qualified": why_not,
            "collected_user_info": collected_info,  # Preserve for reference
        },
    )


if __name__ == "__main__":
    import asyncio

    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)

    # Test result tracking
    test_results = {"passed": 0, "failed": 0, "errors": []}

    def _assert_with_tracking(condition: bool, message: str, test_name: str) -> None:
        """Enhanced assertion with test result tracking."""
        if not condition:
            error_msg = f"[{test_name}] {message}"
            test_results["errors"].append(error_msg)
            test_results["failed"] += 1
            print(f"✗ {error_msg}")
            raise AssertionError(error_msg)
        test_results["passed"] += 1

    async def test_collect_user_info_complete() -> None:
        """Test collecting complete user information."""
        print("\nTesting: Complete user info collection")

        state = QualifierSubgraphState(
            messages=["I'm 25 years old, living in Maryland, ZIP 21201"]
        )
        cmd = await collect_user_info(state)

        _assert_with_tracking(
            cmd.goto == "qualify_user",
            f"Should proceed to qualify_user, got {cmd.goto}",
            "complete_info",
        )

        info = cmd.update.get("collected_user_info")
        _assert_with_tracking(
            isinstance(info, UserInfoOutputSchema),
            f"Expected UserInfoOutputSchema, got {type(info)}",
            "complete_info_type",
        )
        _assert_with_tracking(
            info.age == 25, f"Age should be 25, got {info.age}", "age_25"
        )
        _assert_with_tracking(
            info.state == "Maryland",
            f"State should be Maryland, got {info.state}",
            "state_md",
        )
        print(f"✓ Complete info collection: {info.model_dump()}")

    async def test_collect_user_info_missing_fields() -> None:
        """Test handling of missing required fields."""
        print("\nTesting: Missing fields handling")

        # Missing ZIP
        state = QualifierSubgraphState(messages=["I'm 30 years old"])
        cmd = await collect_user_info(state)

        _assert_with_tracking(
            cmd.goto == "collect_user_info",
            f"Should loop back for missing ZIP, got {cmd.goto}",
            "missing_zip",
        )
        print("✓ Missing ZIP handled correctly")

        # Missing age
        state = QualifierSubgraphState(messages=["My ZIP is 22102"])
        cmd = await collect_user_info(state)

        _assert_with_tracking(
            cmd.goto == "collect_user_info",
            f"Should loop back for missing age, got {cmd.goto}",
            "missing_age",
        )
        print("✓ Missing age handled correctly")

    async def test_collect_user_info_merging() -> None:
        """Test merging of prior information."""
        print("\nTesting: Info merging")

        prior = UserInfoOutputSchema(age=25, state=None, zip_code=None)
        state = QualifierSubgraphState(
            messages=["", "My ZIP is 20001"],  # D.C. ZIP
            collected_user_info=prior,
        )
        cmd = await collect_user_info(state)

        info = cmd.update.get("collected_user_info")
        _assert_with_tracking(
            info.age == 25,
            f"Age should be preserved as 25, got {info.age}",
            "merge_age",
        )
        _assert_with_tracking(
            info.state == "District of Columbia",
            f"State should be inferred as D.C., got {info.state}",
            "merge_dc_state",
        )
        _assert_with_tracking(
            info.zip_code == "20001",
            f"ZIP should be 20001, got {info.zip_code}",
            "merge_zip",
        )
        print(f"✓ Merging works: {info.model_dump()}")

    async def test_qualify_user_qualified() -> None:
        """Test qualification of eligible users."""
        print("\nTesting: Qualified users")

        test_cases = [
            (25, "Maryland", "21201", "MD adult"),
            (18, "Virginia", "22102", "VA exactly 18"),
            (30, "District of Columbia", "20001", "D.C. adult"),
            (50, "District of Columbia", "20500", "D.C. second range"),
        ]

        for age, state, zip_code, description in test_cases:
            state_obj = QualifierSubgraphState(
                collected_user_info=UserInfoOutputSchema(
                    age=age, state=state, zip_code=zip_code
                ),
                messages=[""],
            )
            cmd = await qualify_user(state_obj)

            _assert_with_tracking(
                cmd.goto == END,
                f"Should go to END, got {cmd.goto}",
                f"qual_{description}",
            )
            _assert_with_tracking(
                cmd.update.get("is_user_qualified") is True,
                f"{description} should qualify",
                f"qual_{description}_result",
            )
            _assert_with_tracking(
                cmd.update.get("why_not_qualified") is None,
                "Should have no disqualification reason",
                f"qual_{description}_reason",
            )
            print(f"✓ {description} qualified correctly")

    async def test_qualify_user_not_qualified() -> None:
        """Test disqualification of ineligible users."""
        print("\nTesting: Not qualified users")

        test_cases = [
            (17, "Maryland", "21201", "Minor in MD", "age"),
            (25, "California", "90210", "Adult in CA", "location"),
            (16, "Texas", "75001", "Minor in TX", "both"),
        ]

        for age, state, zip_code, description, reason_type in test_cases:
            state_obj = QualifierSubgraphState(
                collected_user_info=UserInfoOutputSchema(
                    age=age, state=state, zip_code=zip_code
                ),
                messages=[""],
            )
            cmd = await qualify_user(state_obj)

            _assert_with_tracking(
                cmd.goto == END,
                f"Should go to END, got {cmd.goto}",
                f"not_qual_{description}",
            )
            _assert_with_tracking(
                cmd.update.get("is_user_qualified") is False,
                f"{description} should not qualify",
                f"not_qual_{description}_result",
            )
            _assert_with_tracking(
                cmd.update.get("why_not_qualified") is not None,
                "Should have disqualification reason",
                f"not_qual_{description}_reason",
            )
            print(
                f"✓ {description} disqualified correctly: {cmd.update.get('why_not_qualified')}"
            )

    async def test_error_handling() -> None:
        """Test error handling in edge cases."""
        print("\nTesting: Error handling")

        # Empty state
        try:
            state = QualifierSubgraphState(messages=[])
            await collect_user_info(state)
            _assert_with_tracking(
                False, "Should raise ValueError for empty messages", "empty_state"
            )
        except ValueError as e:
            _assert_with_tracking(
                "at least one message" in str(e),
                f"Error message should mention messages, got: {e}",
                "empty_state_error",
            )
            print("✓ Empty state handled correctly")

        # Missing collected_user_info in qualify_user
        state = QualifierSubgraphState(messages=["test"])
        cmd = await qualify_user(state)
        _assert_with_tracking(
            cmd.update.get("is_user_qualified") is False,
            "Should default to not qualified when info missing",
            "missing_info_qualify",
        )
        print("✓ Missing info in qualify_user handled correctly")

    async def test_dc_zip_ranges() -> None:
        """Test D.C. ZIP code range handling."""
        print("\nTesting: D.C. ZIP ranges")

        dc_zips = [
            ("20001", "D.C. first range start"),
            ("20050", "D.C. first range middle"),
            ("20099", "D.C. first range end"),
            ("20201", "D.C. second range start"),
            ("20400", "D.C. second range middle"),
            ("20599", "D.C. second range end"),
        ]

        for zip_code, description in dc_zips:
            state = QualifierSubgraphState(messages=[f"I'm 25, ZIP {zip_code}"])
            cmd = await collect_user_info(state)
            info = cmd.update.get("collected_user_info")

            _assert_with_tracking(
                info.state == "District of Columbia",
                f"{description}: State should be D.C., got {info.state}",
                f"dc_zip_{zip_code}",
            )
            print(f"✓ {description} correctly mapped to D.C.")

    async def main() -> None:
        """Run all comprehensive tests for nodes logic."""
        print("\n" + "=" * 60)
        print("QUALIFIER NODES LOGIC COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        # Run all test suites
        await test_collect_user_info_complete()
        await test_collect_user_info_missing_fields()
        await test_collect_user_info_merging()
        await test_qualify_user_qualified()
        await test_qualify_user_not_qualified()
        await test_error_handling()
        await test_dc_zip_ranges()

        # Report results
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"✓ Passed: {test_results['passed']}")
        print(f"✗ Failed: {test_results['failed']}")
        print(f"Total: {test_results['passed'] + test_results['failed']}")

        if test_results["errors"]:
            print("\nErrors encountered:")
            for error in test_results["errors"][:10]:
                print(f"  - {error}")
            if len(test_results["errors"]) > 10:
                print(f"  ... and {len(test_results['errors']) - 10} more")

        if test_results["failed"] == 0:
            print("\n🎉 All tests passed successfully!")
        else:
            print(f"\n⚠️  {test_results['failed']} tests failed. Review errors above.")
            exit(1)

    asyncio.run(main())
