"""Comprehensive test for concierge workflow with job-seeking scenarios.

uv run test_concierge_comprehensive.py
"""

# %%
import asyncio
import time
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def test_job_seeking_scenario(
    scenario_name: str, messages: list, print_details: bool = True
) -> dict:
    """Test a single job-seeking scenario."""
    async with AsyncSqliteSaver.from_conn_string("in-memory") as saver:
        concierge_graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        results = {
            "scenario": scenario_name,
            "total_requests": 0,
            "successful": 0,
            "agent_selections": 0,
            "task_extractions": 0,
            "profile_complete": False,
            "times": [],
        }

        if print_details:
            print(f"\n{'=' * 60}")
            print(f"SCENARIO: {scenario_name}")
            print(f"{'=' * 60}")

        for i, message in enumerate(messages, 1):
            if print_details:
                print(f"\nMessage {i}: {message[:80]}...")

            start_time = time.time()
            result = await concierge_graph.ainvoke({"messages": messages[:i]}, config)
            elapsed = time.time() - start_time
            results["times"].append(elapsed)

            # Check if this is a job request (contains keywords)
            is_job_request = any(
                keyword in message.lower()
                for keyword in ["find", "search", "look for", "job", "career", "hiring"]
            )

            if is_job_request:
                results["total_requests"] += 1

                if result.get("selected_agent"):
                    results["agent_selections"] += 1
                    if print_details:
                        print(f"  ✓ Agent: {result['selected_agent']}")

                if result.get("task"):
                    results["task_extractions"] += 1
                    if print_details:
                        print(f"  ✓ Task: {result['task'][:60]}...")

                if result.get("final_answer"):
                    results["successful"] += 1
                    if print_details:
                        print(
                            f"  ✓ Answer provided ({len(result['final_answer'])} chars)"
                        )
            elif result.get("direct_response_to_the_user"):
                if print_details:
                    print(
                        f"  → Receptionist: {result['direct_response_to_the_user'][:80]}..."
                    )

        # Check final profile
        final_state = await concierge_graph.aget_state(config)
        if final_state.values.get("user_profile"):
            profile = final_state.values["user_profile"]
            if hasattr(profile, "is_valid"):
                results["profile_complete"] = profile.is_valid

        return results


async def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE JOB-SEEKING TEST SUITE")
    print("=" * 70)

    # Define test scenarios
    scenarios = [
        {
            "name": "Quick Job Search - Software Engineer",
            "messages": [
                "Hi, I'm Alex Chen.",
                "I live at 456 University Ave, Boston, MA. I'm unemployed.",
                "I was an intern at StartupXYZ in Boston.",
                "I want entry-level software developer jobs, willing to relocate.",
                "Find entry-level software developer jobs in Seattle and Austin",
                "Search for virtual career fairs happening this month",
            ],
        },
        {
            "name": "Remote Data Science Search",
            "messages": [
                "Hello, I'm Sarah Johnson.",
                "789 Oak St, Austin, TX. Currently employed.",
                "Senior Data Scientist at DataCorp in Austin.",
                "Looking for remote data science roles, $150k+ salary.",
                "Find remote senior data scientist positions with equity",
                "Look for AI/ML conferences with networking opportunities",
            ],
        },
        {
            "name": "Bootcamp Grad Job Hunt",
            "messages": [
                "I'm Michael Rodriguez.",
                "321 Market St, San Francisco, CA. Unemployed.",
                "Was a Marketing Manager at RetailCo in SF.",
                "Want junior web developer roles, open to contract.",
                "Find companies that hire bootcamp graduates in Bay Area",
                "Search for tech meetups in San Francisco this week",
            ],
        },
    ]

    all_results = []

    for scenario in scenarios:
        results = await test_job_seeking_scenario(
            scenario["name"], scenario["messages"], print_details=True
        )
        all_results.append(results)

    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)

    total_requests = sum(r["total_requests"] for r in all_results)
    total_successful = sum(r["successful"] for r in all_results)
    total_agent_selections = sum(r["agent_selections"] for r in all_results)
    total_task_extractions = sum(r["task_extractions"] for r in all_results)
    profiles_complete = sum(1 for r in all_results if r["profile_complete"])
    all_times = [t for r in all_results for t in r["times"]]
    avg_time = sum(all_times) / len(all_times) if all_times else 0

    print(f"\nScenarios tested: {len(scenarios)}")
    print(f"Total job requests: {total_requests}")
    print(
        f"Successful completions: {total_successful}/{total_requests} ({total_successful / total_requests * 100:.1f}%)"
    )
    print(
        f"Agent selections: {total_agent_selections}/{total_requests} ({total_agent_selections / total_requests * 100:.1f}%)"
    )
    print(
        f"Task extractions: {total_task_extractions}/{total_requests} ({total_task_extractions / total_requests * 100:.1f}%)"
    )
    print(
        f"Complete profiles: {profiles_complete}/{len(scenarios)} ({profiles_complete / len(scenarios) * 100:.1f}%)"
    )
    print(f"Average response time: {avg_time:.2f}s")

    # Performance assessment
    print("\n" + "=" * 70)
    print("🎯 PERFORMANCE ASSESSMENT")
    print("=" * 70)

    success_rate = total_successful / total_requests if total_requests > 0 else 0

    if success_rate >= 0.9:
        print("✅ Success Rate: Excellent (90%+)")
    elif success_rate >= 0.7:
        print("⚠️  Success Rate: Good (70-90%)")
    else:
        print("❌ Success Rate: Needs improvement (<70%)")

    if avg_time <= 3.0:
        print(f"✅ Response Time: Excellent ({avg_time:.2f}s)")
    elif avg_time <= 5.0:
        print(f"⚠️  Response Time: Acceptable ({avg_time:.2f}s)")
    else:
        print(f"❌ Response Time: Slow ({avg_time:.2f}s)")

    if profiles_complete == len(scenarios):
        print("✅ Profile Collection: Perfect")
    elif profiles_complete >= len(scenarios) * 0.8:
        print("⚠️  Profile Collection: Good")
    else:
        print("❌ Profile Collection: Needs improvement")

    # Overall verdict
    print("\n" + "=" * 70)
    if success_rate >= 0.8 and avg_time <= 5.0:
        print("🎉 OVERALL: PASSED! System is working well for job-seeking scenarios.")
    elif success_rate >= 0.6:
        print("⚠️  OVERALL: PASSED with warnings. Some improvements recommended.")
    else:
        print("❌ OVERALL: FAILED. Significant improvements required.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
# %%
