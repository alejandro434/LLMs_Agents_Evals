"""Comprehensive debug test to ensure correctness, reliability, robustness and consistency.

uv run test_comprehensive_debug.py
"""

# %%
import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def test_single_scenario_detailed() -> None:
    """Test a single scenario with detailed debugging."""
    print("\n" + "=" * 70)
    print("DETAILED DEBUG TEST - MARIA'S SCENARIO")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("debug_detailed.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        messages = [
            "hi! so I'm thinking about switching careers",
            "Maria Santos is my name",
            "I'm at 456 Main Street in Richmond, Virginia, 23220",
            "yeah still working but looking to make a change",
            "currently I'm a Sales Manager at RetailChain here in Richmond",
            "I'm interested in maybe project management or operations roles? hybrid would be nice",
        ]

        print("\n📊 Testing State Persistence and Information Extraction")
        print("-" * 60)

        for i, msg in enumerate(messages, 1):
            print(f"\n[Turn {i}] User: {msg}")

            # Invoke the graph
            result = await concierge_graph.ainvoke({"messages": [msg]}, config)

            # Check the state
            state = await concierge_graph.aget_state(config)

            print("\n  State Analysis:")
            print(f"    - Next nodes: {state.next}")
            print(f"    - State keys: {list(state.values.keys())[:5]}")

            # Check if we have user_profile
            if "user_profile" in state.values:
                profile = state.values["user_profile"]
                if profile:
                    print(f"    - User Profile exists: {type(profile)}")
                    if hasattr(profile, "name"):
                        print(f"      • Name: {profile.name}")
                    if hasattr(profile, "employment_status"):
                        print(f"      • Employment: {profile.employment_status}")
                    if hasattr(profile, "last_job"):
                        print(f"      • Job: {profile.last_job}")
                    if hasattr(profile, "last_job_company"):
                        print(f"      • Company: {profile.last_job_company}")
                    if hasattr(profile, "job_preferences"):
                        print(f"      • Preferences: {profile.job_preferences}")

            # Check for handoff
            if result.get("selected_agent"):
                print("\n  ✅ HANDOFF DETECTED!")
                print(f"    - Agent: {result['selected_agent']}")
                print(f"    - Task: {result.get('task', 'None')[:60]}...")
                break

            # Check for response
            if result.get("direct_response_to_the_user"):
                print(
                    f"\n  AI Response: {result['direct_response_to_the_user'][:60]}..."
                )

            # After turn 6, we should have all info
            if i == 6:
                print("\n  ⚠️  ISSUE: Should have handed off by now!")
                print("  Checking what's missing...")

                # Get the latest state
                if state.values.get("user_profile"):
                    profile = state.values["user_profile"]
                    missing = []
                    if not hasattr(profile, "name") or not profile.name:
                        missing.append("name")
                    if (
                        not hasattr(profile, "current_address")
                        or not profile.current_address
                    ):
                        missing.append("address")
                    if (
                        not hasattr(profile, "employment_status")
                        or not profile.employment_status
                    ):
                        missing.append("employment")
                    if not hasattr(profile, "last_job") or not profile.last_job:
                        missing.append("job")
                    if (
                        not hasattr(profile, "last_job_company")
                        or not profile.last_job_company
                    ):
                        missing.append("company")
                    if (
                        not hasattr(profile, "last_job_location")
                        or not profile.last_job_location
                    ):
                        missing.append("location")
                    if (
                        not hasattr(profile, "job_preferences")
                        or not profile.job_preferences
                    ):
                        missing.append("preferences")

                    if missing:
                        print(f"    Missing fields: {missing}")
                    else:
                        print("    All fields present but handoff didn't occur!")


async def test_receptionist_subgraph_directly() -> None:
    """Test the receptionist subgraph in isolation."""
    print("\n" + "=" * 70)
    print("TESTING RECEPTIONIST SUBGRAPH DIRECTLY")
    print("=" * 70)

    from src.graphs.receptionist_subgraph.lgraph_builder import (
        builder as receptionist_builder,
    )

    async with AsyncSqliteSaver.from_conn_string("receptionist_direct.sqlite") as saver:
        receptionist_graph = receptionist_builder.compile(checkpointer=saver)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        messages = [
            "hi! so I'm thinking about switching careers",
            "Maria Santos is my name",
            "I'm at 456 Main Street in Richmond, Virginia, 23220",
            "yeah still working but looking to make a change",
            "currently I'm a Sales Manager at RetailChain here in Richmond",
            "I'm interested in maybe project management or operations roles? hybrid would be nice",
        ]

        for i, msg in enumerate(messages, 1):
            print(f"\n[Turn {i}] User: {msg[:50]}...")

            try:
                result = await receptionist_graph.ainvoke({"messages": [msg]}, config)

                # Check what we got
                if "receptionist_output_schema" in result:
                    output = result["receptionist_output_schema"]
                    if output.user_info_complete:
                        print("  ✅ INFO COMPLETE - Should trigger handoff!")
                        print(f"    Name: {output.user_name}")
                        print(f"    Job: {output.user_last_job}")
                        print(f"    Company: {output.user_last_job_company}")
                        print(f"    Preferences: {output.user_job_preferences}")
                        break
                    else:
                        missing = []
                        if not output.user_name:
                            missing.append("name")
                        if not output.user_current_address:
                            missing.append("address")
                        if not output.user_employment_status:
                            missing.append("employment")
                        if not output.user_last_job:
                            missing.append("job")
                        if not output.user_last_job_company:
                            missing.append("company")
                        if not output.user_last_job_location:
                            missing.append("location")
                        if not output.user_job_preferences:
                            missing.append("preferences")
                        print(f"  Missing: {missing}")

                if "user_profile" in result:
                    print("  User profile created!")

            except Exception as e:
                print(f"  Interrupt or error: {str(e)[:60]}...")


async def test_consistency_multiple_runs() -> None:
    """Test consistency by running the same scenario multiple times."""
    print("\n" + "=" * 70)
    print("TESTING CONSISTENCY - MULTIPLE RUNS")
    print("=" * 70)

    results = []

    for run in range(3):
        print(f"\n--- Run {run + 1} ---")

        async with AsyncSqliteSaver.from_conn_string(
            f"consistency_test_{run}.sqlite"
        ) as saver:
            concierge_graph = builder.compile(checkpointer=saver)
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Use Robert's scenario (which worked before)
            messages = [
                "Hi, need help finding a new position",
                "Robert Chen",
                "2100 Harbor Blvd, Norfolk VA 23501",
                "unemployed unfortunately, got laid off last month",
                "I was a Senior Engineer at DefenseContractor in Norfolk",
                "looking for senior engineering roles, ideally in defense or aerospace sectors",
            ]

            handoff_occurred = False
            handoff_turn = 0

            for i, msg in enumerate(messages, 1):
                result = await concierge_graph.ainvoke({"messages": [msg]}, config)

                if result.get("selected_agent"):
                    handoff_occurred = True
                    handoff_turn = i
                    print(f"  Handoff at turn {i}")
                    break

            results.append(
                {"run": run + 1, "handoff": handoff_occurred, "turn": handoff_turn}
            )

    # Check consistency
    print("\n📊 Consistency Analysis:")
    if all(r["handoff"] for r in results):
        if len(set(r["turn"] for r in results)) == 1:
            print("  ✅ CONSISTENT: All runs handed off at the same turn")
        else:
            print(
                f"  ⚠️  INCONSISTENT: Handoffs at different turns: {[r['turn'] for r in results]}"
            )
    else:
        print("  ❌ INCONSISTENT: Not all runs resulted in handoff")
        for r in results:
            print(f"    Run {r['run']}: {'Handoff' if r['handoff'] else 'No handoff'}")


async def main() -> None:
    """Run all comprehensive tests."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE CORRECTNESS & RELIABILITY TEST")
    print("=" * 70)

    # Test 1: Detailed single scenario
    await test_single_scenario_detailed()

    # Test 2: Receptionist subgraph directly
    await test_receptionist_subgraph_directly()

    # Test 3: Consistency across multiple runs
    await test_consistency_multiple_runs()

    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
