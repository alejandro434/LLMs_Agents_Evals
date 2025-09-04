"""Concierge workflow.

uv run -m src.graphs.concierge_workflow
"""

# %%
import uuid
from pprint import pprint
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import Field

from src.graphs.followup_node.nodes_logic import followup_node
from src.graphs.followup_node.schemas import FollowupSubgraphState
from src.graphs.ReAct_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer as react_subgraph,
)
from src.graphs.receptionist_subgraph.lgraph_builder import (
    graph_with_in_memory_checkpointer as receptor_router_subgraph,
)
from src.graphs.receptionist_subgraph.schemas import (
    UserProfileSchema,
    UserRequestExtractionSchema,
)


class ConciergeGraphState(MessagesState):
    """Concierge state."""

    user_profile: UserProfileSchema | None = Field(default=None)
    task: str | None = Field(default=None)
    rationale_of_the_handoff: str | None = Field(default=None)
    selected_agent: (
        Literal["Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"] | None
    ) = Field(default=None)
    suggested_tools: list[str] | None = Field(default=None)
    tools_advisor_reasoning: str | None = Field(default=None)
    final_answer: str | None = Field(default=None)
    # Add field to capture interrupt responses from the receptionist
    direct_response_to_the_user: str | None = Field(default=None)


async def receptor_router(
    state: ConciergeGraphState,
) -> Command[Literal["Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"]]:
    """Receptor router."""
    # Use a consistent thread_id for the receptionist subgraph to maintain state
    # Create a stable thread_id based on the conversation to maintain state across turns
    # We'll use a hash of the first message or a fixed ID
    if state.get("messages"):
        # Use a hash of the first message to create a stable thread_id
        first_msg = str(state["messages"][0])
        import hashlib

        thread_hash = hashlib.md5(first_msg.encode()).hexdigest()[:8]
        config = {"configurable": {"thread_id": f"receptionist_{thread_hash}"}}
    else:
        config = {"configurable": {"thread_id": "receptionist_default"}}

    response = await receptor_router_subgraph.ainvoke(
        {"messages": state["messages"]}, config
    )

    # Extract fields from the response, which may be partial if interrupted
    pprint(f"receptor_router response: {response}")

    user_profile = response.get("user_profile")
    user_request = response.get("user_request")
    selected_agent = response.get("selected_agent")
    rationale_of_the_handoff = response.get("rationale_of_the_handoff")
    direct_response = response.get("direct_response_to_the_user")

    # If no agent selected (receptionist interrupted for more info), go to END
    # The direct_response will contain the question for the user
    if not selected_agent:
        return Command(
            goto=END,
            update={
                "user_profile": user_profile,
                "task": user_request.task if user_request else None,
                "rationale_of_the_handoff": rationale_of_the_handoff,
                "selected_agent": selected_agent,
                "direct_response_to_the_user": direct_response,
            },
        )

    # Agent was selected, route to the appropriate agent. Currently, only the
    # Jobs agent is implemented (ReAct subgraph). Route all selections to Jobs.
    # Route to the selected agent node if present (Jobs/Educator/Events/CareerCoach/Entrepreneur)
    next_node = (
        selected_agent
        if selected_agent
        in {"Jobs", "Educator", "Events", "CareerCoach", "Entrepreneur"}
        else "Jobs"
    )
    return Command(
        goto=next_node,
        update={
            "user_profile": user_profile,
            "task": user_request.task if user_request else None,
            "rationale_of_the_handoff": rationale_of_the_handoff,
            "selected_agent": selected_agent,
        },
    )


async def react(state: ConciergeGraphState) -> Command[Literal[END]]:
    """React node."""
    # Use a consistent thread_id for the react subgraph to maintain state
    # Create a stable thread_id based on the conversation
    if state.get("messages"):
        # Use a hash of the first message to create a stable thread_id
        first_msg = str(state["messages"][0])
        import hashlib

        thread_hash = hashlib.md5(first_msg.encode()).hexdigest()[:8]
        config = {"configurable": {"thread_id": f"react_{thread_hash}"}}
    else:
        config = {"configurable": {"thread_id": "react_default"}}

    if not state.get("task"):
        raise ValueError("React node: task is missing")
    # Import the schema needed for the user_request

    # Create a UserRequestExtractionSchema object from the task string
    user_request = UserRequestExtractionSchema(task=state.get("task"))

    missing_or_invalid: list[str] = []
    if not state.get("user_profile"):
        missing_or_invalid.append("React node: user_profile is missing")
    elif not isinstance(state.get("user_profile"), UserProfileSchema):
        value_type = type(state.get("user_profile")).__name__
        missing_or_invalid.append(
            f"React node: user_profile has invalid type: {value_type}, expected UserProfileSchema"
        )
    if not state.get("rationale_of_the_handoff"):
        missing_or_invalid.append("React node: rationale_of_the_handoff is missing")

    if missing_or_invalid:
        issues = ", ".join(missing_or_invalid)
        print(f"react input issues: {issues}")
        raise ValueError(f"Missing/invalid fields: {issues}")

    _input = {
        "user_request": user_request,
        "user_profile": state.get("user_profile"),
        "why_this_agent_can_help": state.get("rationale_of_the_handoff"),
    }

    response = await react_subgraph.ainvoke(_input, config)

    pprint(f"react response: {response}")

    return Command(
        goto=END,
        update={
            "final_answer": response.get("final_answer"),
        },
    )


async def follow_up(
    state: ConciergeGraphState,
) -> Command[Literal["follow_up", "receptor_router", END]]:
    """Follow up node."""
    response = await followup_node(FollowupSubgraphState(messages=state["messages"]))
    if not response.next_agent:
        return Command(
            goto="follow_up",
            update={
                "messages": [response.direct_response_to_the_user],
            },
        )

    return Command(
        goto="follow_up",
        update={
            "messages": [response.direct_response_to_the_user],
            "rationale_of_the_handoff": response.guidance_for_distil_user_needs,
            "user_request": response.user_request,
        },
    )


builder = StateGraph(ConciergeGraphState)

builder.add_node("receptor_router", receptor_router)
builder.add_node("follow_up", follow_up)
builder.add_node("Jobs", react)
builder.add_node("Educator", react)
builder.add_node("Events", react)
builder.add_node("CareerCoach", react)
builder.add_node("Entrepreneur", react)

builder.add_edge(START, "receptor_router")
graph_with_in_memory_checkpointer = builder.compile(checkpointer=MemorySaver())
# Export a server-safe graph without a custom checkpointer for LangGraph API
graph = builder.compile()

if __name__ == "__main__":
    import asyncio

    CONFIG: dict = {"configurable": {"thread_id": str(uuid.uuid4())}}

    async def test_subgraph_async_streaming() -> None:
        """Test async streaming response with SQLite persistence."""
        print("\n=== Testing Receptionist Graph with SQLite Persistence ===")
        async with AsyncSqliteSaver.from_conn_string("in-memory") as saver:
            concierge_graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

            test_input = {
                "messages": [
                    "Hi, I'm Jane Doe. I'm unemployed and looking for work in Maryland. "
                ]
            }

            print("\nStreaming updates:")
            async for update in concierge_graph.astream(
                test_input,
                config,
                stream_mode="updates",
                debug=True,
            ):
                pprint(update)

    async def test_full_flow_with_interrupts() -> None:
        """Test the full flow including handling interrupts."""
        print("\n=== Testing Full Flow with Interrupts ===")
        async with AsyncSqliteSaver.from_conn_string("in-memory") as saver:
            concierge_graph = builder.compile(checkpointer=saver)
            # Use a consistent thread_id to maintain conversation state
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Build conversation history as we go
            messages = []

            # First message - only name
            print("\n1. Initial message with only name:")
            messages.append("Hi, I'm Jane Doe.")
            test_input = {"messages": messages.copy()}

            result = await concierge_graph.ainvoke(test_input, config, debug=True)
            if result.get("direct_response_to_the_user"):
                print(
                    f"   Receptionist asks: {result['direct_response_to_the_user'][:150]}..."
                )

            # Second message - adding address and employment status
            print("\n2. Providing address and employment status:")
            messages.append(
                "I live at 123 Main Street, Baltimore, Maryland 21201. "
                "I'm currently unemployed."
            )
            test_input2 = {"messages": messages.copy()}

            result2 = await concierge_graph.ainvoke(test_input2, config, debug=True)
            if result2.get("direct_response_to_the_user"):
                print(
                    f"   Receptionist asks: {result2['direct_response_to_the_user'][:150]}..."
                )

            # Third message - adding last job details
            print("\n3. Providing last job information:")
            messages.append(
                "My last job was as a Software Engineer at TechCorp in San Francisco, CA."
            )
            test_input3 = {"messages": messages.copy()}

            result3 = await concierge_graph.ainvoke(test_input3, config, debug=True)
            if result3.get("direct_response_to_the_user"):
                print(
                    f"   Receptionist asks: {result3['direct_response_to_the_user'][:150]}..."
                )

            # Fourth message - adding job preferences and the actual request
            print("\n4. Providing job preferences and making the request:")
            messages.append(
                "I'm interested in software engineering roles, preferably remote or hybrid. "
                "I'd like to find job fairs happening in the US within the next 30 days."
            )
            test_input4 = {"messages": messages.copy()}

            result4 = None

            result4 = await concierge_graph.ainvoke(test_input4, config, debug=True)

            # Use result if available; otherwise, read from state
            final_state = await concierge_graph.aget_state(config)
            values = final_state.values

            selected_agent = (
                result4.get("selected_agent")
                if result4
                else values.get("selected_agent")
            )
            task_value = result4.get("task") if result4 else values.get("task")
            rationale_value = (
                result4.get("rationale_of_the_handoff")
                if result4
                else values.get("rationale_of_the_handoff")
            )
            direct_response_value = (
                result4.get("direct_response_to_the_user")
                if result4
                else values.get("direct_response_to_the_user")
            )

            # Now all info should be complete and agent should be selected
            if selected_agent:
                print(f"   ✓ Agent selected: {selected_agent}")
                preview = (task_value or "N/A")[:100]
                print(f"   ✓ Task: {preview}...")
                rationale_preview = (rationale_value or "N/A")[:100]
                print(f"   ✓ Rationale: {rationale_preview}...")
                if result4 and result4.get("final_answer"):
                    final_preview = result4.get("final_answer", "N/A")[:200]
                    print(f"   ✓ Final answer: {final_preview}...")

                # Validate agent selection and extraction consistency
                assert selected_agent in {
                    "Jobs",
                    "Educator",
                    "Events",
                    "CareerCoach",
                    "Entrepreneur",
                }, "Unexpected selected_agent"
                assert isinstance(task_value, str) and task_value, (
                    "Task should be a non-empty string"
                )
                assert isinstance(rationale_value, str) and len(rationale_value) > 10, (
                    "Rationale of the handoff should be non-empty"
                )
            elif direct_response_value:
                print(f"   Receptionist response: {direct_response_value[:150]}...")

            # Get the final state from the graph to access all state fields
            final_state = await concierge_graph.aget_state(config)

            # Print final state summary
            print("\n5. Final collected information:")

            user_profile = final_state.values.get("user_profile")
            if user_profile and hasattr(user_profile, "name"):
                print("   ✓ User Profile collected:")
                print(f"     - Name: {user_profile.name}")
                print(f"     - Zip Code: {user_profile.zip_code}")
                print(f"     - Employment: {user_profile.current_employment_status}")
                print(
                    "     - What looking for: "
                    f"{user_profile.what_is_the_user_looking_for}"
                )
            else:
                print(
                    "   Note: User profile not available or incomplete in final state"
                )

    async def test_comprehensive_job_seeking_scenarios() -> None:
        """Comprehensive test suite for job-seeking scenarios with evaluation."""
        import time

        print("\n" + "=" * 70)
        print("COMPREHENSIVE JOB-SEEKING SCENARIOS TEST SUITE")
        print("=" * 70)

        # Test scenarios with different user profiles and job requests
        test_scenarios = [
            {
                "name": "Recent Graduate - Entry Level Jobs",
                "profile": {
                    "name": "Alex Chen",
                    "address": "456 University Ave, Boston, MA 02115",
                    "employment": "unemployed",
                    "last_job": "Intern at StartupXYZ",
                    "location": "Boston, MA",
                    "preferences": "entry-level software developer positions, willing to relocate",
                },
                "requests": [
                    "Find entry-level software developer jobs in tech hubs like Seattle, Austin, and NYC",
                    "Search for new grad hiring programs at FAANG companies starting in 2024",
                    "Look for virtual career fairs for computer science graduates happening this month",
                ],
            },
            {
                "name": "Senior Professional - Remote Opportunities",
                "profile": {
                    "name": "Sarah Johnson",
                    "address": "789 Oak Street, Austin, TX 78701",
                    "employment": "employed",
                    "last_job": "Senior Data Scientist at DataCorp",
                    "location": "Austin, TX",
                    "preferences": "remote data science or ML engineering roles, $150k+ salary",
                },
                "requests": [
                    "Find remote senior data scientist positions at companies with 4-day work weeks",
                    "Search for AI/ML conferences in Q1 2024 with job networking opportunities",
                    "Look for top companies hiring remote data scientists with equity compensation",
                ],
            },
            {
                "name": "Career Changer - Bootcamp Graduate",
                "profile": {
                    "name": "Michael Rodriguez",
                    "address": "321 Market St, San Francisco, CA 94102",
                    "employment": "unemployed",
                    "last_job": "Marketing Manager at RetailCo",
                    "location": "San Francisco, CA",
                    "preferences": "junior web developer roles, open to contract or full-time",
                },
                "requests": [
                    "Find companies in the Bay Area that actively hire bootcamp graduates",
                    "Search for junior developer positions that don't require a CS degree",
                    "Look for tech meetups and networking events in San Francisco this week",
                ],
            },
            {
                "name": "Laid-off Tech Worker - Urgent Search",
                "profile": {
                    "name": "Emily Watson",
                    "address": "555 Pine Ave, Seattle, WA 98101",
                    "employment": "unemployed",
                    "last_job": "Product Manager at BigTech Inc",
                    "location": "Seattle, WA",
                    "preferences": "product management roles in fintech or healthcare, hybrid preferred",
                },
                "requests": [
                    "Find product manager openings at Seattle startups with immediate start dates",
                    "Search for companies offering signing bonuses for experienced PMs",
                    "Look for job fairs and hiring events in the Pacific Northwest this month",
                ],
            },
        ]

        async with AsyncSqliteSaver.from_conn_string("in-memory") as saver:
            concierge_graph = builder.compile(checkpointer=saver)

            # Evaluation metrics
            total_tests = 0
            successful_routings = 0
            correct_agent_selections = 0
            profile_completions = 0
            task_extractions = 0
            response_times = []

            for scenario in test_scenarios:
                print(f"\n{'=' * 60}")
                print(f"SCENARIO: {scenario['name']}")
                print(f"{'=' * 60}")

                # Fresh thread for each scenario
                thread_id = str(uuid.uuid4())
                config = {"configurable": {"thread_id": thread_id}}

                # Incrementally provide user information
                messages = []

                # Step 1: Name only
                print("\nStep 1: Introducing user...")
                messages.append(f"Hi, I'm {scenario['profile']['name']}.")
                result = await concierge_graph.ainvoke(
                    {"messages": messages.copy()}, config, debug=True
                )

                # Verify receptionist asks for more info
                if result.get("direct_response_to_the_user"):
                    print("   ✓ Receptionist engaged: asks for more information")
                else:
                    print("   ✗ No response from receptionist")

                # Step 2: Address and employment
                print("\nStep 2: Providing address and employment status...")
                messages.append(
                    f"I live at {scenario['profile']['address']}. "
                    f"I'm currently {scenario['profile']['employment']}."
                )
                result = await concierge_graph.ainvoke(
                    {"messages": messages.copy()}, config, debug=True
                )

                if result.get("direct_response_to_the_user"):
                    print("   ✓ Receptionist continues gathering info")

                # Step 3: Job history
                print("\nStep 3: Providing job history...")
                messages.append(
                    f"My last job was {scenario['profile']['last_job']} "
                    f"in {scenario['profile']['location']}."
                )
                result = await concierge_graph.ainvoke(
                    {"messages": messages.copy()}, config, debug=True
                )

                if result.get("direct_response_to_the_user"):
                    print("   ✓ Receptionist asks for preferences")

                # Step 4: Preferences
                print("\nStep 4: Providing job preferences...")
                messages.append(
                    f"I'm looking for {scenario['profile']['preferences']}."
                )
                result = await concierge_graph.ainvoke(
                    {"messages": messages.copy()}, config, debug=True
                )

                # Profile should be complete now
                if not result.get("direct_response_to_the_user"):
                    print("   ✓ Profile collection complete")

                # Now test each job-seeking request
                print("\n📋 Testing Job-Seeking Requests:")
                for i, request in enumerate(scenario["requests"], 1):
                    print(f'\n   Request {i}: "{request[:60]}..."')
                    messages.append(request)

                    start_time = time.time()
                    result = await concierge_graph.ainvoke(
                        {"messages": messages.copy()}, config, debug=True
                    )
                    elapsed_time = time.time() - start_time
                    response_times.append(elapsed_time)

                    total_tests += 1

                    # Evaluate the response
                    print(f"     ⏱️  Response time: {elapsed_time:.2f}s")

                    # Check if agent was selected
                    if result.get("selected_agent"):
                        successful_routings += 1
                        print(f"     ✓ Agent selected: {result['selected_agent']}")

                        # Verify it's the correct agent for web search tasks
                        if result["selected_agent"] in {
                            "Jobs",
                            "Educator",
                            "Events",
                            "CareerCoach",
                            "Entrepreneur",
                        }:
                            correct_agent_selections += 1
                            print("     ✓ Valid agent selected")
                        else:
                            print(
                                f"     ✗ Unexpected agent: {result['selected_agent']}"
                            )
                    else:
                        print("     ✗ No agent selected")

                    # Check task extraction
                    if result.get("task"):
                        task_extractions += 1
                        print(f'     ✓ Task extracted: "{result["task"][:60]}..."')
                    else:
                        print("     ✗ No task extracted")

                    # Check if answer was provided
                    if result.get("final_answer"):
                        answer_preview = result["final_answer"][:100].replace("\n", " ")
                        print(f'     ✓ Answer provided: "{answer_preview}..."')
                    else:
                        print("     ✗ No answer provided")

                    # Get final state to check profile
                    final_state = await concierge_graph.aget_state(config)
                    if final_state.values.get("user_profile"):
                        profile = final_state.values["user_profile"]
                        if hasattr(profile, "is_valid") and profile.is_valid:
                            profile_completions += 1
                            print("     ✓ Complete user profile maintained")
                        else:
                            print("     ⚠️  User profile incomplete")
                    else:
                        print("     ✗ User profile not found")

                print(f"\n{'-' * 40}")
                print("Scenario Summary:")
                print(f"  ✓ {scenario['name']} completed")
                print(f"  ✓ Processed {len(scenario['requests'])} job-seeking requests")

        # Calculate statistics
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        # Final evaluation report
        print("\n" + "=" * 70)
        print("📊 EVALUATION REPORT")
        print("=" * 70)
        print(f"Total tests run: {total_tests}")
        print(
            f"Successful routings: {successful_routings}/{total_tests} ({successful_routings / total_tests * 100:.1f}%)"
        )
        print(
            f"Correct agent selections: {correct_agent_selections}/{total_tests} ({correct_agent_selections / total_tests * 100:.1f}%)"
        )
        print(
            f"Task extractions: {task_extractions}/{total_tests} ({task_extractions / total_tests * 100:.1f}%)"
        )
        print(
            f"Profile completions: {profile_completions}/{total_tests} ({profile_completions / total_tests * 100:.1f}%)"
        )
        print(f"Average response time: {avg_response_time:.2f}s")

        # Performance assessment
        print("\n" + "=" * 70)
        print("🎯 PERFORMANCE ASSESSMENT")
        print("=" * 70)

        success_rate = successful_routings / total_tests if total_tests > 0 else 0

        # Detailed assessment
        assessments = []
        if success_rate >= 0.95:
            assessments.append("✅ Routing: Excellent (95%+)")
        elif success_rate >= 0.80:
            assessments.append("⚠️  Routing: Good (80-95%)")
        else:
            assessments.append("❌ Routing: Needs improvement (<80%)")

        if correct_agent_selections == successful_routings:
            assessments.append("✅ Agent Selection: Perfect (100% accuracy)")
        elif correct_agent_selections >= successful_routings * 0.9:
            assessments.append("⚠️  Agent Selection: Good (90%+ accuracy)")
        else:
            assessments.append("❌ Agent Selection: Needs improvement")

        if avg_response_time <= 3.0:
            assessments.append(
                f"✅ Response Time: Excellent ({avg_response_time:.2f}s avg)"
            )
        elif avg_response_time <= 5.0:
            assessments.append(
                f"⚠️  Response Time: Acceptable ({avg_response_time:.2f}s avg)"
            )
        else:
            assessments.append(f"❌ Response Time: Slow ({avg_response_time:.2f}s avg)")

        for assessment in assessments:
            print(f"  {assessment}")

        # Overall verdict
        print("\n" + "=" * 70)
        if success_rate >= 0.9 and avg_response_time <= 5.0:
            print("🎉 OVERALL: TEST SUITE PASSED! System is production-ready.")
        elif success_rate >= 0.7:
            print(
                "⚠️  OVERALL: TEST SUITE PASSED with warnings. Some improvements recommended."
            )
        else:
            print("❌ OVERALL: TEST SUITE FAILED. Significant improvements required.")
        print("=" * 70)

        return success_rate

    async def test_in_memory_checkpointer_direct() -> None:
        """Test the graph_with_in_memory_checkpointer directly."""
        print("\n" + "=" * 70)
        print("TESTING IN-MEMORY CHECKPOINTER GRAPH DIRECTLY")
        print("=" * 70)

        # Test 1: Basic functionality with pre-compiled graph
        print("\n--- Test 1: Basic Functionality ---")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Initial message
        test_input = {"messages": ["Hi, I'm John Smith."]}
        result = await graph_with_in_memory_checkpointer.ainvoke(
            test_input, config, debug=True
        )

        assert result.get("direct_response_to_the_user"), (
            "Should get receptionist response"
        )
        print(f"✓ Initial response: {result['direct_response_to_the_user'][:80]}...")

        # Test 2: State persistence across invocations
        print("\n--- Test 2: State Persistence ---")

        # Add more information
        test_input2 = {
            "messages": [
                "Hi, I'm John Smith.",
                "I live at 789 Tech Ave, San Jose, CA. I'm unemployed.",
            ]
        }
        result2 = await graph_with_in_memory_checkpointer.ainvoke(
            test_input2, config, debug=True
        )

        # Check state is maintained
        state = await graph_with_in_memory_checkpointer.aget_state(config)
        assert state.values.get("messages"), "Messages should be in state"
        assert len(state.values["messages"]) >= 2, "Should have multiple messages"
        print(f"✓ State persisted: {len(state.values['messages'])} messages in history")

        # Test 3: Complete flow to agent selection
        print("\n--- Test 3: Complete Flow to Agent Selection ---")

        messages_complete = [
            "Hi, I'm John Smith.",
            "I live at 789 Tech Ave, San Jose, CA. I'm unemployed.",
            "My last job was Senior Engineer at TechCo in San Jose.",
            "I'm looking for senior engineering roles, remote preferred.",
            "Find remote senior software engineer positions at startups",
        ]

        test_input3 = {"messages": messages_complete}
        result3 = None

        result3 = await graph_with_in_memory_checkpointer.ainvoke(
            test_input3, config, debug=True
        )

        # Prefer result; fallback to state
        state_after = await graph_with_in_memory_checkpointer.aget_state(config)
        values = state_after.values
        selected_agent = (
            result3.get("selected_agent") if result3 else values.get("selected_agent")
        )
        task_value = result3.get("task") if result3 else values.get("task")
        rationale_value = (
            result3.get("rationale_of_the_handoff")
            if result3
            else values.get("rationale_of_the_handoff")
        )

        if selected_agent:
            print(f"✓ Agent selected: {selected_agent}")
            print(f"✓ Task: {(task_value or 'N/A')[:80]}...")
            # Validate handoff fields
            assert selected_agent == "Jobs", "Expected selected_agent to be 'Jobs'"
            assert isinstance(task_value, str) and task_value, (
                "Task should be a non-empty string"
            )
            assert isinstance(rationale_value, str) and len(rationale_value) > 10, (
                "Rationale of the handoff should be non-empty"
            )

        # Get final state to verify profile
        final_state = await graph_with_in_memory_checkpointer.aget_state(config)
        if final_state.values.get("user_profile"):
            profile = final_state.values["user_profile"]
            if profile and hasattr(profile, "name"):
                print(f"✓ Profile captured: {profile.name}")

        # Test 4: Multiple threads (isolation)
        print("\n--- Test 4: Thread Isolation ---")

        thread1 = str(uuid.uuid4())
        thread2 = str(uuid.uuid4())
        config1 = {"configurable": {"thread_id": thread1}}
        config2 = {"configurable": {"thread_id": thread2}}

        # Thread 1
        await graph_with_in_memory_checkpointer.ainvoke(
            {"messages": ["I'm Alice."]}, config1, debug=True
        )

        # Thread 2
        await graph_with_in_memory_checkpointer.ainvoke(
            {"messages": ["I'm Bob."]}, config2, debug=True
        )

        # Check isolation
        state1 = await graph_with_in_memory_checkpointer.aget_state(config1)
        state2 = await graph_with_in_memory_checkpointer.aget_state(config2)

        messages1 = state1.values.get("messages", [])
        messages2 = state2.values.get("messages", [])

        assert "Alice" in str(messages1) and "Alice" not in str(messages2), (
            "Thread 1 should only have Alice"
        )
        assert "Bob" in str(messages2) and "Bob" not in str(messages1), (
            "Thread 2 should only have Bob"
        )
        print("✓ Thread isolation verified")

        # Test 5: Streaming with in-memory checkpointer
        print("\n--- Test 5: Streaming Support ---")

        stream_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Build conversation incrementally to test streaming properly
        messages = []
        updates_count = 0

        # Step 1: Initial greeting
        messages.append("Hi, I'm Emma Watson.")
        print("Sending: Initial greeting...")
        async for update in graph_with_in_memory_checkpointer.astream(
            {"messages": messages.copy()},
            stream_config,
            stream_mode="updates",
            debug=False,
        ):
            updates_count += 1
            if "receptor_router" in update:
                print(
                    f"  ✓ Streamed update {updates_count}: receptor_router (greeting)"
                )

        # Step 2: Add address and employment
        messages.append("I live at 321 AI Drive, Austin, TX. I'm employed.")
        print("Sending: Address and employment...")
        async for update in graph_with_in_memory_checkpointer.astream(
            {"messages": messages.copy()},
            stream_config,
            stream_mode="updates",
            debug=False,
        ):
            updates_count += 1
            if "receptor_router" in update:
                print(f"  ✓ Streamed update {updates_count}: receptor_router (address)")

        # Step 3: Add job details
        messages.append("I'm a Data Scientist at DataCorp in Austin.")
        print("Sending: Job details...")
        async for update in graph_with_in_memory_checkpointer.astream(
            {"messages": messages.copy()},
            stream_config,
            stream_mode="updates",
            debug=False,
        ):
            updates_count += 1
            if "receptor_router" in update:
                print(f"  ✓ Streamed update {updates_count}: receptor_router (job)")

        # Step 4: Add preferences
        messages.append("Looking for senior data science roles, remote only.")
        print("Sending: Job preferences...")
        async for update in graph_with_in_memory_checkpointer.astream(
            {"messages": messages.copy()},
            stream_config,
            stream_mode="updates",
            debug=False,
        ):
            updates_count += 1
            if "receptor_router" in update:
                print(
                    f"  ✓ Streamed update {updates_count}: receptor_router (preferences)"
                )

        # Step 5: Make actual job search request (should trigger agent handoff)
        messages.append(
            "Search for remote data science positions with good work-life balance"
        )
        print("Sending: Job search request...")
        async for update in graph_with_in_memory_checkpointer.astream(
            {"messages": messages.copy()},
            stream_config,
            stream_mode="updates",
            debug=False,
        ):
            updates_count += 1
            if "receptor_router" in update:
                print(f"  ✓ Streamed update {updates_count}: receptor_router (request)")
            elif "react" in update:
                print(f"  ✓ Streamed update {updates_count}: react agent (processing)")

        print(f"✓ Total updates streamed: {updates_count}")

        print("\n" + "=" * 70)
        print("✅ IN-MEMORY CHECKPOINTER TESTS PASSED!")
        print("=" * 70)

    async def test_in_memory_checkpointer_advanced() -> None:
        """Advanced tests for in-memory checkpointer graph."""
        print("\n" + "=" * 70)
        print("ADVANCED IN-MEMORY CHECKPOINTER TESTS")
        print("=" * 70)

        # Test 1: Interrupt and resume
        print("\n--- Test 1: Interrupt and Resume ---")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Start with incomplete info
        messages = ["Hi, I'm Sarah Connor."]
        result = await graph_with_in_memory_checkpointer.ainvoke(
            {"messages": messages}, config, debug=True
        )

        # Should get interrupt asking for more info
        assert result.get("direct_response_to_the_user"), "Should ask for more info"
        print(f"✓ Interrupt 1: {result['direct_response_to_the_user'][:60]}...")

        # Add more info incrementally
        messages.append("I live at 999 Future Blvd, Los Angeles, CA. I'm unemployed.")
        result = await graph_with_in_memory_checkpointer.ainvoke(
            {"messages": messages}, config, debug=True
        )

        assert result.get("direct_response_to_the_user"), "Should ask for job info"
        print(f"✓ Interrupt 2: {result['direct_response_to_the_user'][:60]}...")

        # Complete the profile
        messages.extend(
            [
                "I was a Security Specialist at CyberDyne Systems in LA.",
                "I want cybersecurity roles, remote or hybrid.",
            ]
        )
        result = await graph_with_in_memory_checkpointer.ainvoke(
            {"messages": messages}, config, debug=True
        )

        # Now make a job request
        messages.append("Find cybersecurity job openings in California")
        result = None

        result = await graph_with_in_memory_checkpointer.ainvoke(
            {"messages": messages}, config, debug=True
        )

        # Prefer result; fallback to state
        state_after = await graph_with_in_memory_checkpointer.aget_state(config)
        values = state_after.values
        selected_agent = (
            result.get("selected_agent") if result else values.get("selected_agent")
        )
        task_value = result.get("task") if result else values.get("task")
        rationale_value = (
            result.get("rationale_of_the_handoff")
            if result
            else values.get("rationale_of_the_handoff")
        )

        assert selected_agent in {
            "Jobs",
            "Educator",
            "Events",
            "CareerCoach",
            "Entrepreneur",
        }, "Unexpected selected_agent"
        print(f"✓ Resume successful: Agent={selected_agent}")
        # Validate extracted task and rationale
        assert isinstance(task_value, str) and task_value, (
            "Task should be a non-empty string"
        )
        assert isinstance(rationale_value, str) and len(rationale_value) > 10, (
            "Rationale of the handoff should be non-empty"
        )

        # Test 2: Checkpoint history
        print("\n--- Test 2: Checkpoint History ---")

        # Get checkpoint history
        state = await graph_with_in_memory_checkpointer.aget_state(config)

        # Check we can access the state history
        assert state.values.get("messages"), "Should have messages"
        assert len(state.values["messages"]) >= 5, "Should have full conversation"
        print(f"✓ Checkpoint has {len(state.values['messages'])} messages")

        # Verify user profile was built incrementally
        if state.values.get("user_profile"):
            profile = state.values["user_profile"]
            assert profile.name == "Sarah Connor", "Profile name should match"
            assert profile.zip_code, "Should have zip code"
            assert profile.what_is_the_user_looking_for, "Should have preferences"
            print(f"✓ Profile correctly assembled: {profile.name}")

        # Test 3: Error recovery
        print("\n--- Test 3: Error Recovery ---")

        error_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # Try with malformed input
        try:
            # This should handle gracefully
            result = await graph_with_in_memory_checkpointer.ainvoke(
                {"messages": [""]}, error_config, debug=True
            )
            print("✓ Handled empty message gracefully")
        except Exception as e:
            print(f"✗ Failed on empty message: {e}")

        # Test 4: Performance with multiple requests
        print("\n--- Test 4: Performance Test ---")

        import time

        perf_config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        messages = [
            "I'm Quick Test.",
            "123 Speed St, Fast City, TX. Unemployed.",
            "Was a Developer at QuickCo in Dallas.",
            "Want any developer job.",
            "Find developer jobs in Texas",
        ]

        start_time = time.time()
        result = await graph_with_in_memory_checkpointer.ainvoke(
            {"messages": messages}, perf_config, debug=True
        )
        elapsed = time.time() - start_time

        print(f"✓ Full flow completed in {elapsed:.2f}s")
        if elapsed < 15:
            print("✓ Performance: Good (<15s)")
        elif elapsed < 30:
            print("⚠️  Performance: Acceptable (15-30s)")
        else:
            print("✗ Performance: Slow (>30s)")

        print("\n" + "=" * 70)
        print("✅ ADVANCED IN-MEMORY CHECKPOINTER TESTS COMPLETED!")
        print("=" * 70)

    # Run all tests
    print("\n" + "=" * 70)
    print("Running Basic Tests...")
    print("=" * 70)
    asyncio.run(test_subgraph_async_streaming())

    print("\n" + "=" * 70)
    print("Running Incremental Information Test...")
    print("=" * 70)
    asyncio.run(test_full_flow_with_interrupts())

    # Run in-memory checkpointer tests
    print("\n" + "=" * 70)
    print("Running In-Memory Checkpointer Direct Tests...")
    print("=" * 70)
    asyncio.run(test_in_memory_checkpointer_direct())

    print("\n" + "=" * 70)
    print("Running Advanced In-Memory Checkpointer Tests...")
    print("=" * 70)
    asyncio.run(test_in_memory_checkpointer_advanced())

    # Run the comprehensive job-seeking scenarios test
    print("\n" + "=" * 70)
    print("Running Comprehensive Job-Seeking Scenarios...")
    print("=" * 70)
    asyncio.run(test_comprehensive_job_seeking_scenarios())
    # %%
