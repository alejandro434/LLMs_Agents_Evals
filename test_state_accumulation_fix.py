"""Test and fix state accumulation in the receptionist subgraph.

This script tests that the receptionist properly accumulates state across
multiple conversation turns without losing previously extracted information.

uv run test_state_accumulation_fix.py
"""

# %%
import asyncio
import uuid

from langgraph.checkpoint.memory import MemorySaver

from src.graphs.receptionist_subgraph.lgraph_builder import builder


async def test_maria_scenario_with_subgraph() -> None:
    """Test Maria's exact scenario with the receptionist subgraph."""
    print("\n" + "=" * 70)
    print("TESTING MARIA'S SCENARIO - STATE ACCUMULATION")
    print("=" * 70)

    # Compile the receptionist subgraph with memory
    memory = MemorySaver()
    receptionist_graph = builder.compile(checkpointer=memory)

    # Configuration with thread ID for state persistence
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Maria's conversation steps
    conversation = [
        "hi! so I'm thinking about switching careers",
        "Maria Santos is my name",
        "I'm at 456 Main Street in Richmond, Virginia, 23220",
        "yeah still working but looking to make a change",
        "currently I'm a Sales Manager at RetailChain here in Richmond",
        "I'm interested in maybe project management or operations roles? hybrid would be nice",
    ]

    # Track extracted information
    extracted_fields = {}
    all_fields_extracted = False

    for turn, user_input in enumerate(conversation, 1):
        print(f"\n--- Turn {turn} ---")
        print(f"User: {user_input}")

        # Invoke the subgraph
        try:
            result = await receptionist_graph.ainvoke(
                {"messages": [user_input]},
                config,
            )
        except Exception as e:
            # Handle interrupts (expected for incomplete profiles)
            if "__interrupt__" in str(e) or "interrupt" in str(e).lower():
                # Get the current state to check what was extracted
                state = await receptionist_graph.aget_state(config)
                result = state.values
            else:
                print(f"Error: {e}")
                continue

        # Check the receptionist output
        if "receptionist_output_schema" in result:
            output = result["receptionist_output_schema"]

            # Track what's been extracted
            fields_to_check = {
                "user_name": "Name",
                "user_current_address": "Address",
                "user_employment_status": "Employment",
                "user_last_job": "Job",
                "user_last_job_company": "Company",
                "user_last_job_location": "Location",
                "user_job_preferences": "Preferences",
            }

            print("\n📋 Current Extracted Information:")
            for field, label in fields_to_check.items():
                value = getattr(output, field, None)
                if value:
                    if field not in extracted_fields:
                        print(f"  ✅ NEW: {label}: {value}")
                        extracted_fields[field] = value
                    else:
                        print(f"  ✓ {label}: {value}")

            # Check if all information is complete
            if output.user_info_complete:
                print("\n🎯 USER INFO COMPLETE!")
                all_fields_extracted = True

                # Check for handoff
                if "user_profile" in result:
                    profile = result["user_profile"]
                    print(f"User Profile Created: {profile}")
                if "selected_agent" in result:
                    print(f"Selected Agent: {result['selected_agent']}")
                break

            # Show AI response
            if output.direct_response_to_the_user:
                print(f"\nAI: {output.direct_response_to_the_user[:100]}...")

    # Final verification
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    if all_fields_extracted:
        print("✅ SUCCESS: All fields were extracted and profile was completed!")
    else:
        print("❌ FAILURE: Not all fields were extracted")
        print("\nMissing fields:")
        required = [
            "user_name",
            "user_current_address",
            "user_employment_status",
            "user_last_job",
            "user_last_job_company",
            "user_last_job_location",
            "user_job_preferences",
        ]
        for field in required:
            if field not in extracted_fields:
                print(f"  - {field}")

    # Get final state to verify persistence
    final_state = await receptionist_graph.aget_state(config)
    final_output = final_state.values.get("receptionist_output_schema", {})

    print("\n" + "=" * 70)
    print("FINAL STATE DUMP")
    print("=" * 70)
    print(f"Name: {final_output.user_name}")
    print(f"Address: {final_output.user_current_address}")
    print(f"Employment: {final_output.user_employment_status}")
    print(f"Job: {final_output.user_last_job}")
    print(f"Company: {final_output.user_last_job_company}")
    print(f"Location: {final_output.user_last_job_location}")
    print(f"Preferences: {final_output.user_job_preferences}")
    print(f"Info Complete: {final_output.user_info_complete}")

    return all_fields_extracted


async def test_jake_scenario() -> None:
    """Test Jake's scenario (recent college grad)."""
    print("\n" + "=" * 70)
    print("TESTING JAKE'S SCENARIO - RECENT GRAD")
    print("=" * 70)

    memory = MemorySaver()
    receptionist_graph = builder.compile(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    conversation = [
        "hey there! just graduated and looking for work",
        "oh yeah, I'm Jake Thompson",
        "I live in Baltimore, 789 College Ave, zip is 21218",
        "nope, unemployed right now... just finished school",
        "I did an internship last summer at this startup called TechStartup, here in Baltimore",
        "honestly I'm pretty flexible, looking for entry level tech stuff, could move if needed",
    ]

    for turn, user_input in enumerate(conversation, 1):
        print(f"\n--- Turn {turn} ---")
        print(f"User: {user_input[:50]}...")

        try:
            result = await receptionist_graph.ainvoke(
                {"messages": [user_input]},
                config,
            )
        except Exception as e:
            if "__interrupt__" in str(e) or "interrupt" in str(e).lower():
                state = await receptionist_graph.aget_state(config)
                result = state.values
            else:
                print(f"Error: {e}")
                continue

        if "receptionist_output_schema" in result:
            output = result["receptionist_output_schema"]

            # Quick status check
            extracted = []
            if output.user_name:
                extracted.append(f"name={output.user_name}")
            if output.user_employment_status:
                extracted.append(f"emp={output.user_employment_status}")
            if output.user_last_job:
                extracted.append(f"job={output.user_last_job[:20]}...")
            if output.user_job_preferences:
                extracted.append(f"prefs={output.user_job_preferences[:20]}...")

            if extracted:
                print(f"Extracted: {', '.join(extracted)}")

            if output.user_info_complete:
                print("\n✅ JAKE'S PROFILE COMPLETE!")
                return True

    return False


async def test_robert_scenario() -> None:
    """Test Robert's scenario (experienced professional)."""
    print("\n" + "=" * 70)
    print("TESTING ROBERT'S SCENARIO - EXPERIENCED PRO")
    print("=" * 70)

    memory = MemorySaver()
    receptionist_graph = builder.compile(checkpointer=memory)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    conversation = [
        "Hi, need help finding a new position",
        "Robert Chen",
        "2100 Harbor Blvd, Norfolk VA 23501",
        "unemployed unfortunately, got laid off last month",
        "I was a Senior Engineer at DefenseContractor in Norfolk",
        "looking for senior engineering roles, ideally in defense or aerospace sectors",
    ]

    for turn, user_input in enumerate(conversation, 1):
        print(f"\n--- Turn {turn} ---")
        print(f"User: {user_input[:50]}...")

        try:
            result = await receptionist_graph.ainvoke(
                {"messages": [user_input]},
                config,
            )
        except Exception as e:
            if "__interrupt__" in str(e) or "interrupt" in str(e).lower():
                state = await receptionist_graph.aget_state(config)
                result = state.values
            else:
                print(f"Error: {e}")
                continue

        if "receptionist_output_schema" in result:
            output = result["receptionist_output_schema"]

            # Quick status check
            if output.user_info_complete:
                print("\n✅ ROBERT'S PROFILE COMPLETE!")
                print(f"  Name: {output.user_name}")
                print(f"  Job: {output.user_last_job}")
                print(f"  Company: {output.user_last_job_company}")
                print(f"  Preferences: {output.user_job_preferences}")
                return True

    return False


async def main() -> None:
    """Run all scenario tests."""
    print("\n" + "=" * 70)
    print("STATE ACCUMULATION TEST SUITE")
    print("=" * 70)

    results = {}

    # Test Maria's scenario
    print("\n[TEST 1] Maria - Career Changer")
    results["Maria"] = await test_maria_scenario_with_subgraph()

    # Test Jake's scenario
    print("\n[TEST 2] Jake - Recent Grad")
    results["Jake"] = await test_jake_scenario()

    # Test Robert's scenario
    print("\n[TEST 3] Robert - Experienced Pro")
    results["Robert"] = await test_robert_scenario()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")

    print(f"\nOverall: {success_count}/{total_count} tests passed")

    if success_count == total_count:
        print("\n🎉 ALL TESTS PASSED! State accumulation is working correctly.")
    else:
        print("\n⚠️  Some tests failed. State accumulation needs fixes.")


if __name__ == "__main__":
    asyncio.run(main())
