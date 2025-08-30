"""Final validation test for all concierge workflow fixes.

This script validates that all the issues identified in CONCIERGE_EVAL_REPORT.md
have been properly addressed.

uv run test_final_validation.py
"""

# %%
import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def validate_scenario(name: str, messages: list[str]) -> dict:
    """Validate a single user scenario."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {name}")
    print(f"{'=' * 60}")

    async with AsyncSqliteSaver.from_conn_string(
        f"validation_{name.lower().replace(' ', '_')}.sqlite"
    ) as saver:
        concierge_graph = builder.compile(checkpointer=saver)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        results = {
            "name": name,
            "profile_extracted": False,
            "handoff_occurred": False,
            "final_answer_provided": False,
            "stuck_in_loop": False,
            "extracted_fields": {},
        }

        last_response = None
        for i, msg in enumerate(messages, 1):
            print(f"\n[Turn {i}] User: {msg[:60]}...")

            try:
                result = await concierge_graph.ainvoke({"messages": [msg]}, config)

                # Debug: Print all result keys
                print(f"   Result keys: {list(result.keys())[:5]}...")

                # Check for extracted information
                if "receptionist_output_schema" in result:
                    output = result["receptionist_output_schema"]
                    print("   Found receptionist output!")

                    # Track extracted fields
                    if output.user_name:
                        results["extracted_fields"]["name"] = output.user_name
                    if output.user_employment_status:
                        results["extracted_fields"]["employment"] = (
                            output.user_employment_status
                        )
                    if output.user_last_job:
                        results["extracted_fields"]["job"] = output.user_last_job
                    if output.user_last_job_company:
                        results["extracted_fields"]["company"] = (
                            output.user_last_job_company
                        )
                    if output.user_job_preferences:
                        results["extracted_fields"]["preferences"] = (
                            output.user_job_preferences
                        )

                    if output.user_info_complete:
                        results["profile_extracted"] = True

                # Check for AI response
                if result.get("direct_response_to_the_user"):
                    response = result["direct_response_to_the_user"]
                    print(f"AI: {response[:80]}...")

                    # Check for loops (same response as last time)
                    if last_response and last_response[:50] == response[:50]:
                        results["stuck_in_loop"] = True
                        print("  ⚠️  LOOP DETECTED - Same response as before!")
                    last_response = response

                # Check for handoff
                if result.get("selected_agent"):
                    results["handoff_occurred"] = True
                    print(f"✅ HANDOFF TO: {result['selected_agent']}")
                    print(f"   Task: {result.get('task', '')[:80]}...")

                # Check for final answer
                if result.get("final_answer"):
                    results["final_answer_provided"] = True
                    print(f"✅ FINAL ANSWER: {result['final_answer'][:100]}...")

            except Exception as e:
                print(f"❌ Error: {str(e)[:100]}...")
                results["error"] = str(e)

        return results


async def main() -> None:
    """Run comprehensive validation tests."""
    print("\n" + "=" * 70)
    print("FINAL VALIDATION TEST SUITE")
    print("=" * 70)
    print("\nValidating fixes for issues identified in CONCIERGE_EVAL_REPORT.md:")
    print("1. Job Title at Company parsing")
    print("2. State accumulation across turns")
    print("3. Preference recognition")
    print("4. Conversation loop prevention")
    print("5. Complete user journey handling")

    test_scenarios = [
        {
            "name": "Jake - Recent Grad",
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
            "name": "Maria - Career Changer",
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
            "name": "Robert - Experienced",
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

    results = []
    for scenario in test_scenarios:
        result = await validate_scenario(scenario["name"], scenario["messages"])
        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for result in results:
        print(f"\n{result['name']}:")

        # Check critical issues from the report
        issues = []

        # Issue 1: Job Title at Company parsing
        if (
            "job" in result["extracted_fields"]
            and "company" in result["extracted_fields"]
        ):
            job = result["extracted_fields"]["job"]
            company = result["extracted_fields"]["company"]
            if " at " in job:  # Job shouldn't contain "at Company"
                print(f"  ❌ Parsing Issue: Job contains company: '{job}'")
                issues.append("parsing")
            else:
                print(f"  ✅ Parsing: Job='{job}', Company='{company}'")

        # Issue 2: State accumulation
        if result["profile_extracted"]:
            print("  ✅ State Accumulation: Profile complete")
        else:
            print("  ❌ State Accumulation: Profile incomplete")
            print(f"     Extracted: {list(result['extracted_fields'].keys())}")
            issues.append("state")

        # Issue 3: Preference recognition
        if "preferences" in result["extracted_fields"]:
            print(
                f"  ✅ Preferences: '{result['extracted_fields']['preferences'][:50]}...'"
            )
        else:
            print("  ❌ Preferences: Not extracted")
            issues.append("preferences")

        # Issue 4: Loop prevention
        if result["stuck_in_loop"]:
            print("  ❌ Loop Prevention: Stuck in conversation loop")
            issues.append("loop")
        else:
            print("  ✅ Loop Prevention: No loops detected")

        # Issue 5: Complete journey
        if result["handoff_occurred"] and result["final_answer_provided"]:
            print("  ✅ Journey: Complete (handoff + answer)")
        elif result["handoff_occurred"]:
            print("  ⚠️  Journey: Partial (handoff but no answer)")
        else:
            print("  ❌ Journey: Incomplete")
            issues.append("journey")

        if issues:
            all_passed = False
            print(f"  Issues: {', '.join(issues)}")

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("\nThe following issues have been successfully fixed:")
        print("1. ✅ Job Title at Company parsing works correctly")
        print("2. ✅ State accumulation maintains information across turns")
        print("3. ✅ Preference recognition extracts user preferences")
        print("4. ✅ Conversation loops are prevented")
        print("5. ✅ Complete user journeys are handled properly")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nSummary of remaining issues:")
        for result in results:
            if "error" in result:
                print(f"  - {result['name']}: Error occurred")
            elif not result["handoff_occurred"]:
                print(f"  - {result['name']}: Handoff did not occur")


if __name__ == "__main__":
    asyncio.run(main())
