"""Test script to diagnose and fix receptionist parsing issues.

This script tests the specific parsing failure identified in the evaluation report
where "I'm a Sales Manager at RetailChain" format fails to extract job and company.

uv run test_receptionist_parsing_fix.py
"""

# %%
import asyncio

from src.graphs.receptionist_subgraph.chains import get_receptionist_chain


async def test_job_at_company_parsing() -> None:
    """Test parsing of 'Job Title at Company' format."""
    print("\n" + "=" * 70)
    print("TESTING 'JOB TITLE AT COMPANY' PARSING")
    print("=" * 70)

    # Create the receptionist chain
    receptionist_chain = get_receptionist_chain(temperature=0)

    # Test cases that should work but are failing
    test_cases = [
        {
            "name": "Sales Manager at RetailChain",
            "history": [
                {"human": "Hi, I'm thinking about switching careers"},
                {"ai": "I can help with that! What's your name?"},
                {"human": "Maria Santos is my name"},
                {"ai": "Thanks Maria! Where are you located?"},
                {"human": "I'm at 456 Main Street in Richmond, Virginia, 23220"},
                {"ai": "Great! Are you currently employed?"},
                {"human": "yeah still working but looking to make a change"},
                {"ai": "What's your current role and company?"},
            ],
            "input": "currently I'm a Sales Manager at RetailChain here in Richmond",
            "expected": {
                "user_last_job": "Sales Manager",
                "user_last_job_company": "RetailChain",
                "user_last_job_location": "Richmond",
            },
        },
        {
            "name": "Senior Engineer at DefenseContractor",
            "history": [
                {"human": "Hi, need help finding a new position"},
                {"ai": "I'll help you! What's your name?"},
                {"human": "Robert Chen"},
                {"ai": "Thanks Robert! Where are you located?"},
                {"human": "2100 Harbor Blvd, Norfolk VA 23501"},
                {"ai": "Are you currently employed?"},
                {"human": "unemployed unfortunately, got laid off last month"},
                {"ai": "Sorry to hear that. What was your last position?"},
            ],
            "input": "I was a Senior Engineer at DefenseContractor in Norfolk",
            "expected": {
                "user_last_job": "Senior Engineer",
                "user_last_job_company": "DefenseContractor",
                "user_last_job_location": "Norfolk",
            },
        },
        {
            "name": "Data Scientist at DataCorp",
            "history": [
                {"human": "looking for remote data science roles"},
                {"ai": "I can help! What's your name?"},
                {"human": "Emma Watson"},
                {"ai": "Where are you located?"},
                {"human": "321 AI Drive, Austin, TX"},
                {"ai": "Are you currently employed?"},
                {"human": "yes, employed"},
                {"ai": "What's your current role?"},
            ],
            "input": "I'm a Data Scientist at DataCorp in Austin",
            "expected": {
                "user_last_job": "Data Scientist",
                "user_last_job_company": "DataCorp",
                "user_last_job_location": "Austin",
            },
        },
        {
            "name": "Intern at TechStartup",
            "history": [
                {"human": "just graduated and looking for work"},
                {"ai": "Great! What's your name?"},
                {"human": "Jake Thompson"},
                {"ai": "Where are you located?"},
                {"human": "Baltimore, MD"},
                {"ai": "Are you employed?"},
                {"human": "nope, unemployed"},
                {"ai": "Any recent work experience?"},
            ],
            "input": "I did an internship at TechStartup here in Baltimore",
            "expected": {
                "user_last_job": "Intern",
                "user_last_job_company": "TechStartup",
                "user_last_job_location": "Baltimore",
            },
        },
    ]

    results = []
    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        print(f"Input: '{test_case['input']}'")

        # Invoke the chain with history
        result = await receptionist_chain.ainvoke(
            test_case["input"], current_history=test_case["history"]
        )

        # Check extraction
        success = True
        for field, expected_value in test_case["expected"].items():
            actual_value = getattr(result, field)
            if actual_value is None:
                print(f"  ❌ {field}: None (expected: {expected_value})")
                success = False
            elif expected_value.lower() not in str(actual_value).lower():
                print(f"  ⚠️  {field}: {actual_value} (expected: {expected_value})")
                success = False
            else:
                print(f"  ✅ {field}: {actual_value}")

        results.append(
            {"test": test_case["name"], "success": success, "result": result}
        )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    success_count = sum(1 for r in results if r["success"])
    print(f"Tests passed: {success_count}/{len(test_cases)}")

    if success_count < len(test_cases):
        print("\n⚠️  PARSING ISSUES DETECTED!")
        print("The receptionist chain is failing to extract job information correctly.")
        print("\nFailed tests:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['test']}")
                print(f"    Result: {r['result'].model_dump_json(indent=2)}")

    return results


async def test_preference_recognition() -> None:
    """Test recognition of job preferences in various formats."""
    print("\n" + "=" * 70)
    print("TESTING PREFERENCE RECOGNITION")
    print("=" * 70)

    receptionist_chain = get_receptionist_chain(temperature=0)

    test_cases = [
        {
            "name": "Preferences mixed with question",
            "history": [
                {"human": "I'm Maria Santos"},
                {"ai": "Where are you located?"},
                {"human": "Richmond, VA"},
                {"ai": "Employment status?"},
                {"human": "employed"},
                {"ai": "Current job?"},
                {"human": "Sales Manager at RetailChain"},
                {"ai": "What kind of roles interest you?"},
            ],
            "input": "I'm interested in maybe project management or operations roles? hybrid would be nice",
            "expected_preference": "project management or operations roles, hybrid",
        },
        {
            "name": "Vague preferences",
            "history": [
                {"human": "I'm Jake"},
                {"ai": "Location?"},
                {"human": "Baltimore"},
                {"ai": "Employed?"},
                {"human": "no"},
                {"ai": "Last job?"},
                {"human": "Intern at TechStartup"},
                {"ai": "What are you looking for?"},
            ],
            "input": "honestly I'm pretty flexible, looking for entry level tech stuff, could move if needed",
            "expected_preference": "entry level tech, willing to relocate",
        },
        {
            "name": "Specific preferences",
            "history": [
                {"human": "Robert Chen here"},
                {"ai": "Location?"},
                {"human": "Norfolk, VA"},
                {"ai": "Employment?"},
                {"human": "unemployed"},
                {"ai": "Previous role?"},
                {"human": "Senior Engineer at DefenseContractor"},
                {"ai": "What type of work?"},
            ],
            "input": "looking for senior engineering roles, ideally in defense or aerospace sectors",
            "expected_preference": "senior engineering roles, defense or aerospace",
        },
    ]

    for test_case in test_cases:
        print(f"\n--- Test: {test_case['name']} ---")
        print(f"Input: '{test_case['input']}'")

        result = await receptionist_chain.ainvoke(
            test_case["input"], current_history=test_case["history"]
        )

        if result.user_job_preferences:
            print(f"  ✅ Preferences extracted: {result.user_job_preferences}")
            # Check if key elements are present
            expected_keywords = test_case["expected_preference"].lower().split(",")
            actual_lower = result.user_job_preferences.lower()
            missing = [
                kw.strip() for kw in expected_keywords if kw.strip() not in actual_lower
            ]
            if missing:
                print(f"  ⚠️  Missing keywords: {missing}")
        else:
            print(
                f"  ❌ No preferences extracted (expected: {test_case['expected_preference']})"
            )


async def test_conversation_loop_prevention() -> None:
    """Test that the system doesn't get stuck in loops."""
    print("\n" + "=" * 70)
    print("TESTING CONVERSATION LOOP PREVENTION")
    print("=" * 70)

    receptionist_chain = get_receptionist_chain(temperature=0)

    # Simulate Maria's conversation that gets stuck
    conversation = [
        {"human": "hi! so I'm thinking about switching careers"},
        {"ai": "I can help! What's your name?"},
        {"human": "Maria Santos is my name"},
        {"ai": "Where are you located?"},
        {"human": "I'm at 456 Main Street in Richmond, Virginia, 23220"},
        {"ai": "Are you currently employed?"},
        {"human": "yeah still working but looking to make a change"},
        {"ai": "What's your current role and company?"},
        {"human": "currently I'm a Sales Manager at RetailChain here in Richmond"},
        {"ai": "What kind of roles interest you?"},
        {
            "human": "I'm interested in maybe project management or operations roles? hybrid would be nice"
        },
    ]

    # Build up the conversation
    history = []
    extracted_info = {}

    for i in range(0, len(conversation), 2):
        if i + 1 < len(conversation):
            history.append(conversation[i])
            history.append(conversation[i + 1])

            # Test the next user input
            if i + 2 < len(conversation):
                user_input = conversation[i + 2]["human"]
                print(f"\nTurn {(i // 2) + 2}: User says: '{user_input[:50]}...'")

                result = await receptionist_chain.ainvoke(
                    user_input, current_history=history
                )

                # Track what's been extracted
                if result.user_name and not extracted_info.get("name"):
                    extracted_info["name"] = result.user_name
                    print(f"  ✅ Extracted name: {result.user_name}")

                if result.user_current_address and not extracted_info.get("address"):
                    extracted_info["address"] = result.user_current_address
                    print(
                        f"  ✅ Extracted address: {result.user_current_address[:30]}..."
                    )

                if result.user_employment_status and not extracted_info.get(
                    "employment"
                ):
                    extracted_info["employment"] = result.user_employment_status
                    print(f"  ✅ Extracted employment: {result.user_employment_status}")

                if result.user_last_job and not extracted_info.get("job"):
                    extracted_info["job"] = result.user_last_job
                    print(f"  ✅ Extracted job: {result.user_last_job}")

                if result.user_last_job_company and not extracted_info.get("company"):
                    extracted_info["company"] = result.user_last_job_company
                    print(f"  ✅ Extracted company: {result.user_last_job_company}")

                if result.user_job_preferences and not extracted_info.get(
                    "preferences"
                ):
                    extracted_info["preferences"] = result.user_job_preferences
                    print(f"  ✅ Extracted preferences: {result.user_job_preferences}")

                # Check if all info is complete
                if result.user_info_complete:
                    print("  ✅ USER INFO COMPLETE - Ready for handoff!")
                    break
                elif result.direct_response_to_the_user:
                    # Check if we're asking for info we already have
                    response_lower = result.direct_response_to_the_user.lower()
                    if "name" in response_lower and extracted_info.get("name"):
                        print("  ⚠️  LOOP DETECTED: Asking for name again!")
                    elif "address" in response_lower and extracted_info.get("address"):
                        print("  ⚠️  LOOP DETECTED: Asking for address again!")
                    elif "job" in response_lower and extracted_info.get("job"):
                        print("  ⚠️  LOOP DETECTED: Asking for job again!")
                    else:
                        print(
                            f"  AI response: '{result.direct_response_to_the_user[:50]}...'"
                        )

    print("\n--- Final Extracted Information ---")
    for key, value in extracted_info.items():
        print(f"  {key}: {value}")

    if (
        len(extracted_info) < 6
    ):  # Should have: name, address, employment, job, company, preferences
        print("\n❌ INCOMPLETE EXTRACTION - Missing fields!")
        missing = ["name", "address", "employment", "job", "company", "preferences"]
        for field in missing:
            if field not in extracted_info:
                print(f"  - Missing: {field}")


async def main() -> None:
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RECEPTIONIST PARSING DIAGNOSTIC TESTS")
    print("=" * 70)

    # Test 1: Job at Company parsing
    print("\n[TEST 1] Job Title at Company Parsing")
    parsing_results = await test_job_at_company_parsing()

    # Test 2: Preference recognition
    print("\n[TEST 2] Preference Recognition")
    await test_preference_recognition()

    # Test 3: Loop prevention
    print("\n[TEST 3] Conversation Loop Prevention")
    await test_conversation_loop_prevention()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    # Provide recommendations
    if any(not r["success"] for r in parsing_results):
        print("\n📋 RECOMMENDATIONS:")
        print(
            "1. The receptionist chain is failing to parse 'Job Title at Company' format"
        )
        print("2. Need to add more explicit examples in fewshots.yml")
        print(
            "3. May need to adjust the system prompt to be more explicit about extraction"
        )
        print("4. Consider adding a post-processing step to handle common patterns")


if __name__ == "__main__":
    asyncio.run(main())
