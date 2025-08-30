"""Evaluation script for concierge workflow with job-seeking users.

uv run -m src.graphs.concierge_eval
"""

# %%
import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def evaluate_job_seekers() -> None:
    """Evaluate the concierge workflow with 3 hypothetical job seekers."""
    print("\n" + "=" * 70)
    print("JOB SEEKER EVALUATION - CONVERSATIONAL QUALITY TEST")
    print("=" * 70)

    # Define 3 realistic job seekers with informal, natural conversation styles
    test_users = [
        {
            "name": "User 1: Recent College Grad",
            "profile": {
                "name": "Jake Thompson",
                "address": "789 College Ave, Baltimore, MD 21218",
                "employment": "unemployed",
                "last_job": "Intern at TechStartup",
                "location": "Baltimore, MD",
                "preferences": "entry level tech jobs, willing to relocate",
            },
            "conversations": [
                # Natural progression of a real conversation
                "hey there! just graduated and looking for work",
                "oh yeah, I'm Jake Thompson",
                "I live in Baltimore, 789 College Ave, zip is 21218",
                "nope, unemployed right now... just finished school",
                "I did an internship last summer at this startup called TechStartup, here in Baltimore",
                "honestly I'm pretty flexible, looking for entry level tech stuff, could move if needed",
                # Actual job search requests
                "can you find me any tech job fairs happening soon? like in the next couple weeks maybe?",
                "also what companies are hiring new grads right now? especially for software roles",
                "are there any remote junior developer positions available? that would be ideal",
            ],
        },
        {
            "name": "User 2: Career Changer",
            "profile": {
                "name": "Maria Santos",
                "address": "456 Main Street, Richmond, VA 23220",
                "employment": "employed",
                "last_job": "Sales Manager at RetailChain",
                "location": "Richmond, VA",
                "preferences": "project management or operations, prefer hybrid",
            },
            "conversations": [
                # More casual, uncertain tone
                "hi! so I'm thinking about switching careers",
                "Maria Santos is my name",
                "I'm at 456 Main Street in Richmond, Virginia, 23220",
                "yeah still working but looking to make a change",
                "currently I'm a Sales Manager at RetailChain here in Richmond",
                "I'm interested in maybe project management or operations roles? hybrid would be nice",
                # Job search requests with uncertainty
                "what project manager jobs are out there that don't require tons of experience?",
                "can you look up which companies offer good training programs for career changers?",
                "also curious about salary ranges for PM roles in Virginia... is 80k realistic?",
            ],
        },
        {
            "name": "User 3: Experienced Professional",
            "profile": {
                "name": "Robert Chen",
                "address": "2100 Harbor Blvd, Norfolk, VA 23501",
                "employment": "unemployed",
                "last_job": "Senior Engineer at DefenseContractor",
                "location": "Norfolk, VA",
                "preferences": "senior engineering roles, defense or aerospace preferred",
            },
            "conversations": [
                # Direct but still conversational
                "Hi, need help finding a new position",
                "Robert Chen",
                "2100 Harbor Blvd, Norfolk VA 23501",
                "unemployed unfortunately, got laid off last month",
                "I was a Senior Engineer at DefenseContractor in Norfolk",
                "looking for senior engineering roles, ideally in defense or aerospace sectors",
                # Specific job search needs
                "what defense contractors are currently hiring in the Virginia area?",
                "can you find any aerospace engineering jobs that require security clearance?",
                "also check if there are any job fairs for veterans and defense professionals coming up",
            ],
        },
    ]

    async with AsyncSqliteSaver.from_conn_string("checkpoints_test.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        # Track metrics
        total_interactions = 0
        successful_completions = 0
        natural_responses = 0
        helpful_responses = 0

        for user in test_users:
            print(f"\n{'=' * 60}")
            print(f"Testing: {user['name']}")
            print(f"{'=' * 60}")

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            messages = []
            profile_complete = False
            job_requests_handled = 0

            # Process each conversation turn
            for i, user_message in enumerate(user["conversations"], 1):
                print(f"\n[Turn {i}] User: {user_message}")
                messages.append(user_message)

                try:
                    # Pass only the latest message - the graph maintains history internally
                    result = await concierge_graph.ainvoke(
                        {"messages": [user_message]}, config
                    )
                except Exception as e:
                    print(f"  ⚠️  Error occurred: {str(e)[:100]}...")
                    result = {"error": str(e)}
                total_interactions += 1

                # Evaluate the response
                if result.get("direct_response_to_the_user"):
                    response = result["direct_response_to_the_user"]
                    print(f"Agent: {response[:150]}...")

                    # Check for natural conversation flow
                    if any(
                        phrase in response.lower()
                        for phrase in [
                            "happy to help",
                            "i can help",
                            "let me",
                            "i'll",
                            "sounds good",
                            "great",
                            "thanks for",
                        ]
                    ):
                        natural_responses += 1
                        print("  ✓ Natural conversational tone")

                    # Check if response is helpful/actionable
                    if any(
                        phrase in response.lower()
                        for phrase in [
                            "i can",
                            "i'll",
                            "let me find",
                            "here are",
                            "you might want",
                            "i recommend",
                            "check out",
                        ]
                    ):
                        helpful_responses += 1
                        print("  ✓ Helpful/actionable response")

                elif result.get("selected_agent"):
                    print(f"Agent selected: {result['selected_agent']}")
                    profile_complete = True

                    if result.get("task"):
                        print(f"Task: {result['task'][:100]}...")

                    if result.get("final_answer"):
                        answer = result["final_answer"]
                        print(f"Final answer: {answer[:200]}...")
                        job_requests_handled += 1

                        # Check if answer addresses the request
                        if any(
                            keyword in answer.lower()
                            for keyword in [
                                "found",
                                "here",
                                "check",
                                "available",
                                "hiring",
                            ]
                        ):
                            helpful_responses += 1
                            print("  ✓ Result-oriented answer provided")

            # Summary for this user
            if profile_complete and job_requests_handled > 0:
                successful_completions += 1
                print("\n✅ User journey completed successfully")
                print("   - Profile collected: Yes")
                print(f"   - Job requests handled: {job_requests_handled}")
            else:
                print("\n⚠️  User journey incomplete")
                print(f"   - Profile collected: {profile_complete}")
                print(f"   - Job requests handled: {job_requests_handled}")

        # Final evaluation metrics
        print("\n" + "=" * 70)
        print("📊 EVALUATION METRICS")
        print("=" * 70)
        print(f"Total interactions: {total_interactions}")
        print(
            f"Successful completions: {successful_completions}/{len(test_users)} users"
        )
        print(
            f"Natural responses: {natural_responses}/{total_interactions} "
            f"({natural_responses / total_interactions * 100:.1f}%)"
        )
        print(
            f"Helpful responses: {helpful_responses}/{total_interactions} "
            f"({helpful_responses / total_interactions * 100:.1f}%)"
        )

        # Quality assessment
        print("\n" + "=" * 70)
        print("🎯 CONVERSATIONAL QUALITY ASSESSMENT")
        print("=" * 70)

        naturalness_score = (
            natural_responses / total_interactions if total_interactions > 0 else 0
        )
        helpfulness_score = (
            helpful_responses / total_interactions if total_interactions > 0 else 0
        )
        completion_rate = successful_completions / len(test_users) if test_users else 0

        if naturalness_score >= 0.7:
            print("✅ Naturalness: Excellent - Agent sounds conversational")
        elif naturalness_score >= 0.5:
            print("⚠️  Naturalness: Good - Some improvements needed")
        else:
            print("❌ Naturalness: Poor - Too robotic/formal")

        if helpfulness_score >= 0.7:
            print("✅ Helpfulness: Excellent - Provides actionable assistance")
        elif helpfulness_score >= 0.5:
            print("⚠️  Helpfulness: Good - Could be more proactive")
        else:
            print("❌ Helpfulness: Poor - Not result-oriented enough")

        if completion_rate >= 0.8:
            print("✅ Task Completion: Excellent - Handles full user journeys")
        elif completion_rate >= 0.6:
            print("⚠️  Task Completion: Good - Some gaps in handling")
        else:
            print("❌ Task Completion: Poor - Fails to complete user requests")

        overall_score = (naturalness_score + helpfulness_score + completion_rate) / 3
        print(f"\n📈 Overall Score: {overall_score * 100:.1f}%")

        if overall_score >= 0.7:
            print(
                "🎉 PASS: Agent provides natural, helpful, result-oriented assistance!"
            )
        elif overall_score >= 0.5:
            print("⚠️  PARTIAL PASS: Agent works but needs conversational improvements")
        else:
            print(
                "❌ FAIL: Agent needs significant improvements in conversation quality"
            )

        print("=" * 70)

        return overall_score


if __name__ == "__main__":
    score = asyncio.run(evaluate_job_seekers())
    print(f"\nFinal evaluation score: {score * 100:.1f}%")
