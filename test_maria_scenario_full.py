"""Test the full Maria scenario from the evaluation report.

This script tests the complete concierge workflow with Maria's exact conversation
to identify where the extraction fails in the full graph context.

uv run test_maria_scenario_full.py
"""

# %%
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from src.graphs.concierge_workflow import builder as concierge_builder


async def test_maria_full_scenario() -> None:
    """Test Maria's complete scenario through the full concierge graph."""
    print("\n" + "=" * 70)
    print("TESTING MARIA'S FULL SCENARIO IN CONCIERGE GRAPH")
    print("=" * 70)

    # Build the graph with memory
    memory = MemorySaver()
    graph = concierge_builder.compile(checkpointer=memory)

    # Maria's conversation from the eval report
    conversation = [
        "hi! so I'm thinking about switching careers",
        "Maria Santos is my name",
        "I'm at 456 Main Street in Richmond, Virginia, 23220",
        "yeah still working but looking to make a change",
        "currently I'm a Sales Manager at RetailChain here in Richmond",
        "I'm interested in maybe project management or operations roles? hybrid would be nice",
    ]

    # Track the conversation
    config = {"configurable": {"thread_id": "maria_test"}}
    state = None

    for i, user_input in enumerate(conversation, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_input}")

        # Invoke the graph
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # Get the last AI message
        ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
        if ai_messages:
            last_ai = ai_messages[-1].content
            print(f"AI: {last_ai[:100]}{'...' if len(last_ai) > 100 else ''}")

        # Check the state for extracted information
        if "receptionist_output_schema" in result:
            output = result["receptionist_output_schema"]
            print("\n📋 Extracted Information:")
            if output.user_name:
                print(f"  ✅ Name: {output.user_name}")
            if output.user_current_address:
                print(f"  ✅ Address: {output.user_current_address}")
            if output.user_employment_status:
                print(f"  ✅ Employment: {output.user_employment_status}")
            if output.user_last_job:
                print(f"  ✅ Job: {output.user_last_job}")
            if output.user_last_job_company:
                print(f"  ✅ Company: {output.user_last_job_company}")
            if output.user_last_job_location:
                print(f"  ✅ Location: {output.user_last_job_location}")
            if output.user_job_preferences:
                print(f"  ✅ Preferences: {output.user_job_preferences}")

            # Check if ready for handoff
            if output.user_info_complete:
                print("\n🎯 USER INFO COMPLETE - Ready for handoff!")
                if "selected_agent" in result:
                    print(f"  Selected Agent: {result['selected_agent']}")
                break

        state = result

    # Final check
    if state:
        print("\n" + "=" * 70)
        print("FINAL STATE ANALYSIS")
        print("=" * 70)

        if "receptionist_output_schema" in state:
            output = state["receptionist_output_schema"]
            missing_fields = []

            if not output.user_name:
                missing_fields.append("name")
            if not output.user_current_address:
                missing_fields.append("address")
            if not output.user_employment_status:
                missing_fields.append("employment_status")
            if not output.user_last_job:
                missing_fields.append("last_job")
            if not output.user_last_job_company:
                missing_fields.append("last_job_company")
            if not output.user_last_job_location:
                missing_fields.append("last_job_location")
            if not output.user_job_preferences:
                missing_fields.append("job_preferences")

            if missing_fields:
                print(f"❌ MISSING FIELDS: {', '.join(missing_fields)}")
                print("\nThis explains why the handoff doesn't occur!")
            else:
                print("✅ All fields extracted successfully!")

            # Check for the specific parsing issue
            if output.user_last_job == "Sales Manager at RetailChain":
                print("\n⚠️  PARSING ISSUE DETECTED!")
                print("The job title and company were not separated correctly.")
                print(f"  Got: '{output.user_last_job}'")
                print("  Expected job: 'Sales Manager'")
                print("  Expected company: 'RetailChain'")


async def test_with_direct_receptionist() -> None:
    """Test Maria's scenario directly with the receptionist subgraph."""
    print("\n" + "=" * 70)
    print("TESTING DIRECTLY WITH RECEPTIONIST SUBGRAPH")
    print("=" * 70)

    from src.graphs.receptionist_subgraph.lgraph_builder import (
        build_receptionist_subgraph,
    )

    # Build just the receptionist subgraph
    receptionist_graph = build_receptionist_subgraph()

    # Compile with memory
    memory = MemorySaver()
    app = receptionist_graph.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "maria_receptionist_test"}}

    # Maria's conversation
    conversation = [
        "hi! so I'm thinking about switching careers",
        "Maria Santos is my name",
        "I'm at 456 Main Street in Richmond, Virginia, 23220",
        "yeah still working but looking to make a change",
        "currently I'm a Sales Manager at RetailChain here in Richmond",
        "I'm interested in maybe project management or operations roles? hybrid would be nice",
    ]

    for i, user_input in enumerate(conversation, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_input}")

        # Invoke the subgraph
        result = await app.ainvoke(
            {"messages": [user_input]},
            config=config,
        )

        # Check what was extracted
        if "receptionist_output_schema" in result:
            output = result["receptionist_output_schema"]
            if output.direct_response_to_the_user:
                print(f"AI: {output.direct_response_to_the_user[:80]}...")

            # Show extracted fields
            extracted = []
            if output.user_name:
                extracted.append(f"name={output.user_name}")
            if output.user_employment_status:
                extracted.append(f"employment={output.user_employment_status}")
            if output.user_last_job:
                extracted.append(f"job={output.user_last_job}")
            if output.user_last_job_company:
                extracted.append(f"company={output.user_last_job_company}")
            if output.user_job_preferences:
                extracted.append(f"prefs={output.user_job_preferences[:30]}...")

            if extracted:
                print(f"Extracted: {', '.join(extracted)}")

            # Check if complete
            if output.user_info_complete:
                print("\n✅ INFO COMPLETE - Handoff should occur!")
                break

    # Final analysis
    if "receptionist_output_schema" in result:
        output = result["receptionist_output_schema"]
        print("\n" + "=" * 70)
        print("FINAL EXTRACTION RESULTS")
        print("=" * 70)
        print(f"Name: {output.user_name}")
        print(f"Address: {output.user_current_address}")
        print(f"Employment: {output.user_employment_status}")
        print(f"Job: {output.user_last_job}")
        print(f"Company: {output.user_last_job_company}")
        print(f"Location: {output.user_last_job_location}")
        print(f"Preferences: {output.user_job_preferences}")
        print(f"\nInfo Complete: {output.user_info_complete}")


async def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MARIA SCENARIO FULL TESTING")
    print("=" * 70)

    # Test 1: Full concierge graph
    print("\n[TEST 1] Full Concierge Graph")
    try:
        await test_maria_full_scenario()
    except Exception as e:
        print(f"❌ Error in full scenario: {e}")
        import traceback

        traceback.print_exc()

    # Test 2: Direct receptionist subgraph
    print("\n[TEST 2] Direct Receptionist Subgraph")
    try:
        await test_with_direct_receptionist()
    except Exception as e:
        print(f"❌ Error in receptionist test: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
