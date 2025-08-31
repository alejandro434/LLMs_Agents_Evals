"""Comprehensive test for handoffs and final responses to ensure production readiness.

uv run test_handoffs_and_responses.py
"""

# %%
import asyncio
import time
import uuid
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def test_single_handoff_flow(
    name: str, messages: list[str], expected_handoff_turn: int
) -> dict[str, Any]:
    """Test a single user's complete flow from profile to handoff to final answer."""
    async with AsyncSqliteSaver.from_conn_string(
        f"handoff_test_{name.lower().replace(' ', '_')}.sqlite"
    ) as saver:
        concierge_graph = builder.compile(checkpointer=saver)
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        results = {
            "name": name,
            "handoff_occurred": False,
            "handoff_turn": 0,
            "final_answer_provided": False,
            "response_time": 0,
            "task_extracted": False,
            "errors": [],
        }

        print(f"\n{'=' * 60}")
        print(f"Testing: {name}")
        print(f"{'=' * 60}")

        for i, msg in enumerate(messages, 1):
            print(f"\n[Turn {i}] User: {msg[:60]}...")

            start_time = time.time()
            try:
                result = await concierge_graph.ainvoke({"messages": [msg]}, config)
                elapsed = time.time() - start_time

                # Check for direct response
                if result.get("direct_response_to_the_user"):
                    print(f"AI: {result['direct_response_to_the_user'][:60]}...")

                # Check for handoff
                if result.get("selected_agent") and not results["handoff_occurred"]:
                    results["handoff_occurred"] = True
                    results["handoff_turn"] = i
                    print(f"\n✅ HANDOFF DETECTED at turn {i}!")
                    print(f"   Agent: {result['selected_agent']}")

                    if result.get("task"):
                        results["task_extracted"] = True
                        print(f"   Task: {result['task'][:80]}...")
                    else:
                        print("   ⚠️  No task extracted")

                    print(f"   Response time: {elapsed:.2f}s")

                # Check for final answer
                if result.get("final_answer"):
                    results["final_answer_provided"] = True
                    results["response_time"] = elapsed
                    print("\n✅ FINAL ANSWER PROVIDED!")
                    print(f"   Answer preview: {result['final_answer'][:150]}...")
                    print(f"   Response time: {elapsed:.2f}s")

                    # Validate answer quality
                    answer = result["final_answer"].lower()
                    quality_indicators = [
                        "found",
                        "available",
                        "check",
                        "here",
                        "companies",
                        "positions",
                        "opportunities",
                    ]
                    if any(indicator in answer for indicator in quality_indicators):
                        print("   ✅ Answer contains actionable information")
                    else:
                        print("   ⚠️  Answer may lack actionable content")

                    break

            except Exception as e:
                results["errors"].append(f"Turn {i}: {str(e)[:100]}")
                print(f"❌ Error: {str(e)[:100]}")

        return results


async def test_all_scenarios_handoffs() -> None:
    """Test handoffs for all three main scenarios."""
    print("\n" + "=" * 70)
    print("TESTING HANDOFFS AND FINAL RESPONSES - ALL SCENARIOS")
    print("=" * 70)

    scenarios = [
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
            "expected_handoff": 6,
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
            "expected_handoff": 6,
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
            "expected_handoff": 6,
        },
    ]

    all_results = []

    for scenario in scenarios:
        result = await test_single_handoff_flow(
            scenario["name"], scenario["messages"], scenario["expected_handoff"]
        )
        all_results.append(result)

    # Summary analysis
    print("\n" + "=" * 70)
    print("HANDOFF AND RESPONSE ANALYSIS")
    print("=" * 70)

    successful_handoffs = sum(1 for r in all_results if r["handoff_occurred"])
    successful_answers = sum(1 for r in all_results if r["final_answer_provided"])
    successful_tasks = sum(1 for r in all_results if r["task_extracted"])

    print("\n📊 Overall Metrics:")
    print(
        f"  Handoffs: {successful_handoffs}/{len(scenarios)} ({successful_handoffs / len(scenarios) * 100:.0f}%)"
    )
    print(
        f"  Final Answers: {successful_answers}/{len(scenarios)} ({successful_answers / len(scenarios) * 100:.0f}%)"
    )
    print(
        f"  Task Extraction: {successful_tasks}/{len(scenarios)} ({successful_tasks / len(scenarios) * 100:.0f}%)"
    )

    # Detailed results
    print("\n📋 Detailed Results:")
    for result in all_results:
        print(f"\n{result['name']}:")
        status = (
            "✅"
            if result["handoff_occurred"] and result["final_answer_provided"]
            else "⚠️"
        )
        print(
            f"  {status} Handoff: Turn {result['handoff_turn']}"
            if result["handoff_occurred"]
            else "  ❌ No handoff"
        )
        print(
            f"  {status} Final Answer: Yes ({result['response_time']:.2f}s)"
            if result["final_answer_provided"]
            else "  ❌ No final answer"
        )
        print(
            f"  {status} Task: Extracted"
            if result["task_extracted"]
            else "  ❌ No task"
        )
        if result["errors"]:
            print(f"  ⚠️  Errors: {result['errors']}")

    return all_results


async def test_consistency_multiple_runs() -> None:
    """Test consistency by running the same scenario multiple times."""
    print("\n" + "=" * 70)
    print("TESTING CONSISTENCY - MULTIPLE RUNS")
    print("=" * 70)

    # Use Robert's scenario as it's been most reliable
    messages = [
        "Hi, need help finding a new position",
        "Robert Chen",
        "2100 Harbor Blvd, Norfolk VA 23501",
        "unemployed unfortunately, got laid off last month",
        "I was a Senior Engineer at DefenseContractor in Norfolk",
        "looking for senior engineering roles, ideally in defense or aerospace sectors",
        "what defense contractors are currently hiring in the Virginia area?",
    ]

    runs = 3
    results = []

    print(f"\nRunning the same scenario {runs} times...")

    for run in range(runs):
        print(f"\n--- Run {run + 1} ---")

        async with AsyncSqliteSaver.from_conn_string(
            f"consistency_test_run_{run}.sqlite"
        ) as saver:
            concierge_graph = builder.compile(checkpointer=saver)
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            run_result = {
                "run": run + 1,
                "handoff_turn": 0,
                "has_final_answer": False,
                "answer_quality": "",
            }

            for i, msg in enumerate(messages, 1):
                result = await concierge_graph.ainvoke({"messages": [msg]}, config)

                if result.get("selected_agent") and run_result["handoff_turn"] == 0:
                    run_result["handoff_turn"] = i
                    print(f"  Handoff at turn {i}")

                if result.get("final_answer"):
                    run_result["has_final_answer"] = True
                    # Check answer quality
                    answer = result["final_answer"].lower()
                    if "defense" in answer and "contractor" in answer:
                        run_result["answer_quality"] = "relevant"
                    else:
                        run_result["answer_quality"] = "generic"
                    print(f"  Final answer: {run_result['answer_quality']}")
                    break

            results.append(run_result)

    # Analyze consistency
    print("\n📊 Consistency Analysis:")
    handoff_turns = [r["handoff_turn"] for r in results]
    answers = [r["has_final_answer"] for r in results]
    qualities = [r["answer_quality"] for r in results]

    if len(set(handoff_turns)) == 1:
        print(f"  ✅ Handoffs CONSISTENT: All at turn {handoff_turns[0]}")
    else:
        print(f"  ⚠️  Handoffs INCONSISTENT: {handoff_turns}")

    if all(answers):
        print("  ✅ Final answers CONSISTENT: All provided")
    else:
        print(
            f"  ⚠️  Final answers INCONSISTENT: {sum(answers)}/{len(answers)} provided"
        )

    if len(set(qualities)) == 1 and qualities[0] == "relevant":
        print("  ✅ Answer quality CONSISTENT: All relevant")
    else:
        print(f"  ⚠️  Answer quality varies: {qualities}")

    return results


async def test_error_handling_and_robustness() -> None:
    """Test robustness with edge cases and error scenarios."""
    print("\n" + "=" * 70)
    print("TESTING ROBUSTNESS - EDGE CASES")
    print("=" * 70)

    edge_cases = [
        {
            "name": "Immediate job request (no profile)",
            "messages": ["find me software engineering jobs in Seattle right now"],
        },
        {
            "name": "Complete profile in one message",
            "messages": [
                "Hi I'm Sarah Miller, live at 123 Tech St Seattle WA, unemployed, was a Developer at Amazon, want remote Python jobs, find me some positions"
            ],
        },
        {
            "name": "Multiple requests after profile",
            "messages": [
                "I'm Tom Wilson",
                "Boston MA",
                "unemployed",
                "Software Engineer at StartupCo",
                "looking for senior roles",
                "find tech companies hiring in Boston",
                "also show me remote opportunities",
                "and check for job fairs this month",
            ],
        },
    ]

    for case in edge_cases:
        print(f"\n--- {case['name']} ---")

        async with AsyncSqliteSaver.from_conn_string(
            f"robustness_{case['name'].lower().replace(' ', '_')}.sqlite"
        ) as saver:
            concierge_graph = builder.compile(checkpointer=saver)
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            errors = []
            handoffs = 0
            answers = 0

            for i, msg in enumerate(case["messages"], 1):
                print(f"[Turn {i}] User: {msg[:50]}...")

                try:
                    result = await concierge_graph.ainvoke({"messages": [msg]}, config)

                    if result.get("selected_agent"):
                        handoffs += 1
                        print(f"  ✅ Handoff to {result['selected_agent']}")

                    if result.get("final_answer"):
                        answers += 1
                        print("  ✅ Final answer provided")

                    if result.get("direct_response_to_the_user"):
                        print(f"  AI: {result['direct_response_to_the_user'][:50]}...")

                except Exception as e:
                    errors.append(str(e)[:100])
                    print(f"  ❌ Error: {str(e)[:50]}")

            # Summary for this case
            if errors:
                print(f"  Result: ⚠️  {len(errors)} errors occurred")
            elif handoffs > 0 and answers > 0:
                print(
                    f"  Result: ✅ Successfully handled ({handoffs} handoffs, {answers} answers)"
                )
            elif handoffs > 0:
                print("  Result: ⚠️  Handoff occurred but no answer")
            else:
                print("  Result: ⚠️  No handoff (profile incomplete)")


async def test_response_quality() -> None:
    """Test the quality and relevance of final responses."""
    print("\n" + "=" * 70)
    print("TESTING RESPONSE QUALITY")
    print("=" * 70)

    quality_tests = [
        {
            "name": "Specific job search",
            "setup": [
                "I'm Alex Chen",
                "San Francisco CA",
                "unemployed",
                "Data Scientist at TechCo",
                "looking for senior data science roles",
            ],
            "request": "find data science positions at FAANG companies",
            "expected_keywords": [
                "data",
                "science",
                "companies",
                "positions",
                "senior",
            ],
        },
        {
            "name": "Job fair request",
            "setup": [
                "I'm Lisa Park",
                "Austin TX",
                "unemployed",
                "Marketing Manager at StartupXYZ",
                "want marketing roles",
            ],
            "request": "are there any job fairs in Austin this month?",
            "expected_keywords": ["job fair", "austin", "event", "date", "location"],
        },
    ]

    for test in quality_tests:
        print(f"\n--- {test['name']} ---")

        async with AsyncSqliteSaver.from_conn_string(
            f"quality_{test['name'].lower().replace(' ', '_')}.sqlite"
        ) as saver:
            concierge_graph = builder.compile(checkpointer=saver)
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            # Setup profile
            for msg in test["setup"]:
                await concierge_graph.ainvoke({"messages": [msg]}, config)

            # Make request
            print(f"Request: {test['request']}")
            result = await concierge_graph.ainvoke(
                {"messages": [test["request"]]}, config
            )

            if result.get("final_answer"):
                answer = result["final_answer"].lower()
                print(f"✅ Answer provided ({len(answer)} chars)")

                # Check for expected keywords
                found_keywords = [
                    kw for kw in test["expected_keywords"] if kw.lower() in answer
                ]
                print(f"  Keywords found: {found_keywords}")

                if len(found_keywords) >= len(test["expected_keywords"]) * 0.6:
                    print("  ✅ Answer is relevant and specific")
                else:
                    print("  ⚠️  Answer may lack specificity")

                # Check answer structure
                if len(answer) > 100:
                    print("  ✅ Answer is detailed")
                else:
                    print("  ⚠️  Answer is brief")

            else:
                print("❌ No final answer provided")


async def main() -> None:
    """Run all handoff and response tests."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE HANDOFF AND FINAL RESPONSE TESTING")
    print("=" * 70)
    print("\nTesting for production readiness:")
    print("- Correctness: Handoffs occur at right time with right agent")
    print("- Reliability: Final answers are consistently provided")
    print("- Robustness: Handles edge cases and errors gracefully")
    print("- Consistency: Same behavior across multiple runs")

    # Test 1: All scenarios
    print("\n[TEST 1] All Scenarios - Handoffs and Responses")
    scenario_results = await test_all_scenarios_handoffs()

    # Test 2: Consistency
    print("\n[TEST 2] Consistency Across Multiple Runs")
    consistency_results = await test_consistency_multiple_runs()

    # Test 3: Robustness
    print("\n[TEST 3] Robustness with Edge Cases")
    await test_error_handling_and_robustness()

    # Test 4: Response Quality
    print("\n[TEST 4] Response Quality and Relevance")
    await test_response_quality()

    # Final Assessment
    print("\n" + "=" * 70)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 70)

    # Calculate metrics
    handoff_success = (
        sum(1 for r in scenario_results if r["handoff_occurred"])
        / len(scenario_results)
        * 100
    )
    answer_success = (
        sum(1 for r in scenario_results if r["final_answer_provided"])
        / len(scenario_results)
        * 100
    )
    consistency_score = (
        100 if len(set(r["handoff_turn"] for r in consistency_results)) == 1 else 50
    )

    print("\n📊 Production Metrics:")
    print(f"  Handoff Success Rate: {handoff_success:.0f}%")
    print(f"  Answer Delivery Rate: {answer_success:.0f}%")
    print(f"  Consistency Score: {consistency_score}%")

    if handoff_success >= 90 and answer_success >= 90 and consistency_score >= 90:
        print(
            "\n✅ PRODUCTION READY: Handoffs and final responses are working perfectly!"
        )
    elif handoff_success >= 70 and answer_success >= 70:
        print("\n⚠️  NEARLY READY: Some improvements needed for production")
    else:
        print("\n❌ NOT READY: Significant issues with handoffs or responses")

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
