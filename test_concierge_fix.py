"""Test fix for the concierge evaluation issue.

This script tests the fix for the conversation loop issue where the agent
keeps asking for the same information repeatedly.

uv run test_concierge_fix.py
"""

# %%
import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def test_maria_with_proper_invocation() -> None:
    """Test Maria's scenario with proper message handling."""
    print("\n" + "=" * 70)
    print("TESTING MARIA WITH PROPER MESSAGE HANDLING")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("test_fix.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        conversation = [
            "hi! so I'm thinking about switching careers",
            "Maria Santos is my name",
            "I'm at 456 Main Street in Richmond, Virginia, 23220",
            "yeah still working but looking to make a change",
            "currently I'm a Sales Manager at RetailChain here in Richmond",
            "I'm interested in maybe project management or operations roles? hybrid would be nice",
            "what project manager jobs are out there that don't require tons of experience?",
        ]

        print("\n--- APPROACH 1: Passing only the latest message (CORRECT) ---")
        for i, user_message in enumerate(conversation, 1):
            print(f"\n[Turn {i}] User: {user_message[:60]}...")

            try:
                # Pass only the latest message - the graph maintains history internally
                result = await concierge_graph.ainvoke(
                    {"messages": [user_message]}, config
                )

                # Check what happened
                if result.get("direct_response_to_the_user"):
                    response = result["direct_response_to_the_user"]
                    print(f"AI: {response[:80]}...")
                elif result.get("final_answer"):
                    print(f"Final answer: {result['final_answer'][:80]}...")

                # Check if profile was completed
                if result.get("selected_agent"):
                    print(f"✅ Handoff to: {result['selected_agent']}")
                    print(f"Task: {result.get('task', '')[:80]}...")
                    break

            except Exception as e:
                print(f"Error: {str(e)[:100]}...")


async def test_comparison_approaches() -> None:
    """Compare different approaches to invoking the graph."""
    print("\n" + "=" * 70)
    print("COMPARING INVOCATION APPROACHES")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("test_comparison.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        # Test case: Simple profile collection
        test_messages = [
            "Hi, I need help finding a job",
            "My name is Test User",
            "I live in Baltimore, MD",
        ]

        # Approach 1: Accumulating messages (WRONG - causes loops)
        print("\n--- APPROACH 1: Accumulating messages (WRONG) ---")
        thread_id = str(uuid.uuid4())
        config1 = {"configurable": {"thread_id": thread_id}}
        accumulated_messages = []

        for msg in test_messages[:2]:  # Just test first 2 to avoid spam
            accumulated_messages.append(msg)
            print(
                f"\nSending {len(accumulated_messages)} messages: {accumulated_messages}"
            )

            try:
                result = await concierge_graph.ainvoke(
                    {"messages": accumulated_messages.copy()}, config1
                )
                if result.get("direct_response_to_the_user"):
                    print(f"Response: {result['direct_response_to_the_user'][:60]}...")
            except Exception as e:
                print(f"Error: {str(e)[:60]}...")

        # Approach 2: Single message per turn (CORRECT)
        print("\n--- APPROACH 2: Single message per turn (CORRECT) ---")
        thread_id = str(uuid.uuid4())
        config2 = {"configurable": {"thread_id": thread_id}}

        for msg in test_messages:
            print(f"\nSending single message: '{msg}'")

            try:
                result = await concierge_graph.ainvoke({"messages": [msg]}, config2)
                if result.get("direct_response_to_the_user"):
                    print(f"Response: {result['direct_response_to_the_user'][:60]}...")
            except Exception as e:
                print(f"Error: {str(e)[:60]}...")


async def test_all_scenarios_with_fix() -> None:
    """Test all three scenarios with the proper invocation."""
    print("\n" + "=" * 70)
    print("TESTING ALL SCENARIOS WITH FIX")
    print("=" * 70)

    test_users = [
        {
            "name": "Jake (Recent Grad)",
            "messages": [
                "hey there! just graduated and looking for work",
                "oh yeah, I'm Jake Thompson",
                "I live in Baltimore, 789 College Ave, zip is 21218",
                "nope, unemployed right now... just finished school",
                "I did an internship last summer at this startup called TechStartup, here in Baltimore",
                "honestly I'm pretty flexible, looking for entry level tech stuff, could move if needed",
                "can you find me any tech job fairs happening soon?",
            ],
        },
        {
            "name": "Maria (Career Changer)",
            "messages": [
                "hi! so I'm thinking about switching careers",
                "Maria Santos is my name",
                "I'm at 456 Main Street in Richmond, Virginia, 23220",
                "yeah still working but looking to make a change",
                "currently I'm a Sales Manager at RetailChain here in Richmond",
                "I'm interested in maybe project management or operations roles? hybrid would be nice",
                "what project manager jobs are out there that don't require tons of experience?",
            ],
        },
        {
            "name": "Robert (Experienced)",
            "messages": [
                "Hi, need help finding a new position",
                "Robert Chen",
                "2100 Harbor Blvd, Norfolk VA 23501",
                "unemployed unfortunately, got laid off last month",
                "I was a Senior Engineer at DefenseContractor in Norfolk",
                "looking for senior engineering roles, ideally in defense or aerospace sectors",
                "what defense contractors are currently hiring in the Virginia area?",
            ],
        },
    ]

    async with AsyncSqliteSaver.from_conn_string("test_all_fix.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        for user in test_users:
            print(f"\n{'=' * 60}")
            print(f"Testing: {user['name']}")
            print(f"{'=' * 60}")

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            profile_complete = False
            handoff_occurred = False

            for i, msg in enumerate(user["messages"], 1):
                print(f"\n[Turn {i}] User: {msg[:50]}...")

                try:
                    # Use single message invocation
                    result = await concierge_graph.ainvoke({"messages": [msg]}, config)

                    if result.get("direct_response_to_the_user"):
                        print(f"AI: {result['direct_response_to_the_user'][:60]}...")

                    if result.get("selected_agent"):
                        handoff_occurred = True
                        print("\n✅ HANDOFF SUCCESSFUL!")
                        print(f"   Agent: {result['selected_agent']}")
                        print(f"   Task: {result.get('task', '')[:80]}...")

                    if result.get("final_answer"):
                        print("\n✅ FINAL ANSWER PROVIDED!")
                        print(f"   {result['final_answer'][:150]}...")
                        profile_complete = True
                        break

                except Exception as e:
                    print(f"Error: {str(e)[:100]}...")

            # Summary for this user
            if profile_complete or handoff_occurred:
                print(f"\n✅ {user['name']}: SUCCESS - Profile collected and processed")
            else:
                print(f"\n❌ {user['name']}: FAILED - Stuck in loop")


async def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CONCIERGE FIX TESTING SUITE")
    print("=" * 70)

    # Test 1: Maria with proper invocation
    await test_maria_with_proper_invocation()

    # Test 2: Compare approaches
    await test_comparison_approaches()

    # Test 3: All scenarios with fix
    await test_all_scenarios_with_fix()

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
    print("\n📝 KEY FINDING:")
    print("The issue is in concierge_eval.py line 125:")
    print("  WRONG: {'messages': messages.copy()} - sends entire history")
    print("  RIGHT: {'messages': [user_message]} - sends only latest message")
    print("\nThe graph maintains conversation history internally via checkpointer.")


if __name__ == "__main__":
    asyncio.run(main())
