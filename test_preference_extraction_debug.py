"""Debug preference extraction for Jake and Maria scenarios.

uv run test_preference_extraction_debug.py
"""

# %%
import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def test_jake_preference_extraction() -> None:
    """Test Jake's preference extraction specifically."""
    print("\n" + "=" * 70)
    print("TESTING JAKE'S PREFERENCE EXTRACTION")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("test_jake_prefs.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Jake's conversation up to preferences
        messages = [
            "hey there! just graduated and looking for work",
            "oh yeah, I'm Jake Thompson",
            "I live in Baltimore, 789 College Ave, zip is 21218",
            "nope, unemployed right now... just finished school",
            "I did an internship last summer at this startup called TechStartup, here in Baltimore",
            "honestly I'm pretty flexible, looking for entry level tech stuff, could move if needed",  # PREFERENCES HERE
        ]

        for i, msg in enumerate(messages, 1):
            print(f"\n[Turn {i}] User: {msg[:60]}...")

            result = await concierge_graph.ainvoke({"messages": [msg]}, config)

            # Check what was extracted
            if "receptionist_output_schema" in result:
                output = result["receptionist_output_schema"]
                print("\n📋 Extracted Information:")
                if output.user_name:
                    print(f"  Name: {output.user_name}")
                if output.user_employment_status:
                    print(f"  Employment: {output.user_employment_status}")
                if output.user_last_job:
                    print(f"  Job: {output.user_last_job}")
                if output.user_last_job_company:
                    print(f"  Company: {output.user_last_job_company}")
                if output.user_job_preferences:
                    print(f"  ✅ PREFERENCES: {output.user_job_preferences}")
                else:
                    print("  ❌ NO PREFERENCES EXTRACTED")

                if output.user_info_complete:
                    print("\n✅ INFO COMPLETE - Should handoff!")
                    break

            if result.get("direct_response_to_the_user"):
                print(f"\nAI: {result['direct_response_to_the_user'][:80]}...")

            if result.get("selected_agent"):
                print(f"\n✅ HANDOFF TO: {result['selected_agent']}")
                break


async def test_maria_preference_extraction() -> None:
    """Test Maria's preference extraction specifically."""
    print("\n" + "=" * 70)
    print("TESTING MARIA'S PREFERENCE EXTRACTION")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("test_maria_prefs.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Maria's conversation up to preferences
        messages = [
            "hi! so I'm thinking about switching careers",
            "Maria Santos is my name",
            "I'm at 456 Main Street in Richmond, Virginia, 23220",
            "yeah still working but looking to make a change",
            "currently I'm a Sales Manager at RetailChain here in Richmond",
            "I'm interested in maybe project management or operations roles? hybrid would be nice",  # PREFERENCES HERE
        ]

        for i, msg in enumerate(messages, 1):
            print(f"\n[Turn {i}] User: {msg[:60]}...")

            result = await concierge_graph.ainvoke({"messages": [msg]}, config)

            # Check what was extracted
            if "receptionist_output_schema" in result:
                output = result["receptionist_output_schema"]
                print("\n📋 Extracted Information:")
                if output.user_name:
                    print(f"  Name: {output.user_name}")
                if output.user_employment_status:
                    print(f"  Employment: {output.user_employment_status}")
                if output.user_last_job:
                    print(f"  Job: {output.user_last_job}")
                if output.user_last_job_company:
                    print(f"  Company: {output.user_last_job_company}")
                if output.user_job_preferences:
                    print(f"  ✅ PREFERENCES: {output.user_job_preferences}")
                else:
                    print("  ❌ NO PREFERENCES EXTRACTED")

                if output.user_info_complete:
                    print("\n✅ INFO COMPLETE - Should handoff!")
                    break

            if result.get("direct_response_to_the_user"):
                print(f"\nAI: {result['direct_response_to_the_user'][:80]}...")

            if result.get("selected_agent"):
                print(f"\n✅ HANDOFF TO: {result['selected_agent']}")
                break


async def test_direct_preference_messages() -> None:
    """Test preference extraction with direct messages."""
    print("\n" + "=" * 70)
    print("TESTING DIRECT PREFERENCE MESSAGES")
    print("=" * 70)

    from src.graphs.receptionist_subgraph.chains import get_receptionist_chain

    receptionist_chain = get_receptionist_chain(temperature=0)

    test_cases = [
        {
            "name": "Jake's preferences",
            "history": [
                {"human": "hey there! just graduated and looking for work"},
                {"ai": "What's your name?"},
                {"human": "Jake Thompson"},
                {"ai": "Where are you located?"},
                {"human": "Baltimore"},
                {"ai": "Employment status?"},
                {"human": "unemployed"},
                {"ai": "Last job?"},
                {"human": "Intern at TechStartup"},
                {"ai": "What kind of roles are you looking for?"},
            ],
            "input": "honestly I'm pretty flexible, looking for entry level tech stuff, could move if needed",
        },
        {
            "name": "Maria's preferences",
            "history": [
                {"human": "switching careers"},
                {"ai": "Name?"},
                {"human": "Maria Santos"},
                {"ai": "Location?"},
                {"human": "Richmond, VA"},
                {"ai": "Employment?"},
                {"human": "employed"},
                {"ai": "Current job?"},
                {"human": "Sales Manager at RetailChain"},
                {"ai": "What kind of roles interest you?"},
            ],
            "input": "I'm interested in maybe project management or operations roles? hybrid would be nice",
        },
    ]

    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Input: '{test_case['input']}'")

        result = await receptionist_chain.ainvoke(
            test_case["input"], current_history=test_case["history"]
        )

        if result.user_job_preferences:
            print(f"✅ Preferences extracted: {result.user_job_preferences}")
        else:
            print("❌ No preferences extracted")
            print(f"Full result: {result.model_dump_json(indent=2)}")


async def main() -> None:
    """Run all tests."""
    await test_jake_preference_extraction()
    await test_maria_preference_extraction()
    await test_direct_preference_messages()


if __name__ == "__main__":
    asyncio.run(main())
