"""Debug script to understand why Maria and Robert scenarios are failing.

uv run debug_maria_robert.py
"""

# %%
import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def debug_maria_scenario() -> None:
    """Debug Maria's scenario step by step."""
    print("\n" + "=" * 70)
    print("DEBUGGING MARIA'S SCENARIO")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("debug_maria.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Maria's specific problematic messages
        messages = [
            "hi! so I'm thinking about switching careers",
            "Maria Santos is my name",
            "I'm at 456 Main Street in Richmond, Virginia, 23220",
            "yeah still working but looking to make a change",
            "currently I'm a Sales Manager at RetailChain here in Richmond",  # This is where it gets stuck
        ]

        for i, msg in enumerate(messages, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {msg}")

            try:
                result = await concierge_graph.ainvoke({"messages": [msg]}, config)

                # Print the full state to debug
                print("\n📋 Full Result Keys:", list(result.keys()))

                # Check receptionist output
                if "receptionist_output_schema" in result:
                    output = result["receptionist_output_schema"]
                    print("\n🔍 Receptionist Output:")
                    print(f"  user_name: {output.user_name}")
                    print(f"  user_current_address: {output.user_current_address}")
                    print(f"  user_employment_status: {output.user_employment_status}")
                    print(f"  user_last_job: {output.user_last_job}")
                    print(f"  user_last_job_company: {output.user_last_job_company}")
                    print(f"  user_last_job_location: {output.user_last_job_location}")
                    print(f"  user_job_preferences: {output.user_job_preferences}")
                    print(f"  user_info_complete: {output.user_info_complete}")

                # Check direct response
                if result.get("direct_response_to_the_user"):
                    print(
                        f"\nAI Response: {result['direct_response_to_the_user'][:100]}..."
                    )

                # Check if handoff occurred
                if result.get("selected_agent"):
                    print(f"\n✅ HANDOFF TO: {result['selected_agent']}")
                    print(f"Task: {result.get('task', '')[:100]}...")

                # Get the actual state from the graph
                state = await concierge_graph.aget_state(config)
                print(f"\n🔍 Graph State Next Node: {state.next}")

                # Check if we're stuck in a loop
                if i == 5 and result.get("direct_response_to_the_user"):
                    if "job title" in result["direct_response_to_the_user"].lower():
                        print("\n⚠️  STUCK IN LOOP - Still asking for job title!")

            except Exception as e:
                print(f"Error: {e}")
                import traceback

                traceback.print_exc()


async def debug_robert_scenario() -> None:
    """Debug Robert's scenario step by step."""
    print("\n" + "=" * 70)
    print("DEBUGGING ROBERT'S SCENARIO")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("debug_robert.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        messages = [
            "Hi, need help finding a new position",
            "Robert Chen",
            "2100 Harbor Blvd, Norfolk VA 23501",
            "unemployed unfortunately, got laid off last month",
            "I was a Senior Engineer at DefenseContractor in Norfolk",  # This should extract job info
        ]

        for i, msg in enumerate(messages, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {msg}")

            try:
                result = await concierge_graph.ainvoke({"messages": [msg]}, config)

                # Check if extraction is happening
                if "receptionist_output_schema" in result:
                    output = result["receptionist_output_schema"]
                    print("\n🔍 Extracted Info:")
                    if output.user_name:
                        print(f"  ✓ Name: {output.user_name}")
                    if output.user_employment_status:
                        print(f"  ✓ Employment: {output.user_employment_status}")
                    if output.user_last_job:
                        print(f"  ✓ Job: {output.user_last_job}")
                    if output.user_last_job_company:
                        print(f"  ✓ Company: {output.user_last_job_company}")
                    if output.user_last_job_location:
                        print(f"  ✓ Location: {output.user_last_job_location}")

                    if output.user_info_complete:
                        print("\n✅ INFO COMPLETE!")
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
                        print(f"\n❌ Missing: {', '.join(missing)}")

                if result.get("direct_response_to_the_user"):
                    print(f"\nAI: {result['direct_response_to_the_user'][:100]}...")

            except Exception as e:
                print(f"Error: {e}")


async def main() -> None:
    """Run debug tests."""
    await debug_maria_scenario()
    await debug_robert_scenario()


if __name__ == "__main__":
    asyncio.run(main())
