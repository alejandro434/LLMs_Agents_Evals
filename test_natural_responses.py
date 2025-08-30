"""Test to verify natural, varied responses after fixing the robotic "Thanks for sharing" pattern.

uv run test_natural_responses.py
"""

# %%
import asyncio
import uuid

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.graphs.concierge_workflow import builder


async def test_response_variety() -> None:
    """Test that responses are varied and natural, not robotic."""
    print("\n" + "=" * 70)
    print("TESTING NATURAL RESPONSE VARIETY")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("natural_responses_test.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)

        # Test all three scenarios
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
                ],
            },
        ]

        for scenario in test_scenarios:
            print(f"\n{'=' * 60}")
            print(f"Testing: {scenario['name']}")
            print(f"{'=' * 60}")

            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            responses = []
            thanks_count = 0
            repeated_patterns = {}

            for i, msg in enumerate(scenario["messages"], 1):
                print(f"\n[Turn {i}] User: {msg[:50]}...")

                result = await concierge_graph.ainvoke({"messages": [msg]}, config)

                if result.get("direct_response_to_the_user"):
                    response = result["direct_response_to_the_user"]
                    responses.append(response)
                    
                    # Show the opening of the response
                    opening = response[:60]
                    print(f"AI: {opening}...")
                    
                    # Check for "Thanks for sharing" pattern
                    if "thanks for sharing" in response.lower():
                        thanks_count += 1
                        print("  ⚠️  Contains 'Thanks for sharing'")
                    
                    # Check for repeated opening patterns
                    first_words = response.split()[0:3] if response else []
                    pattern = " ".join(first_words).lower()
                    if pattern in repeated_patterns:
                        repeated_patterns[pattern] += 1
                        if repeated_patterns[pattern] > 1:
                            print(f"  ⚠️  Repeated pattern: '{pattern}' (used {repeated_patterns[pattern]} times)")
                    else:
                        repeated_patterns[pattern] = 1
                    
                    # Check for variety indicators
                    variety_phrases = ["got it", "i see", "perfect", "great", "okay", "alright", "good to know"]
                    if any(phrase in response.lower() for phrase in variety_phrases):
                        print("  ✅ Uses varied acknowledgment")

                if result.get("selected_agent"):
                    print(f"\n✅ Handoff to {result['selected_agent']}")
                    break

            # Analysis for this scenario
            print(f"\n📊 Analysis for {scenario['name']}:")
            print(f"  - Total responses: {len(responses)}")
            print(f"  - 'Thanks for sharing' count: {thanks_count}")
            
            if thanks_count == 0:
                print("  ✅ EXCELLENT: No 'Thanks for sharing' pattern!")
            elif thanks_count == 1:
                print("  ✅ GOOD: Used 'Thanks for sharing' only once")
            elif thanks_count <= 2:
                print("  ⚠️  ACCEPTABLE: Used 'Thanks for sharing' twice")
            else:
                print(f"  ❌ POOR: Overused 'Thanks for sharing' ({thanks_count} times)")
            
            # Check for most repeated pattern
            if repeated_patterns:
                most_repeated = max(repeated_patterns.items(), key=lambda x: x[1])
                if most_repeated[1] > 2:
                    print(f"  ⚠️  Most repeated opening: '{most_repeated[0]}' ({most_repeated[1]} times)")
                else:
                    print("  ✅ Good variety in response openings")


async def test_conversation_naturalness() -> None:
    """Test the overall naturalness of the conversation."""
    print("\n" + "=" * 70)
    print("TESTING CONVERSATION NATURALNESS")
    print("=" * 70)

    async with AsyncSqliteSaver.from_conn_string("naturalness_test.sqlite") as saver:
        concierge_graph = builder.compile(checkpointer=saver)
        
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        # Complete conversation flow
        conversation = [
            ("hi! I need help finding a job", "greeting"),
            ("My name is Alex Johnson", "name"),
            ("I live in Seattle, WA", "location"),
            ("I'm currently unemployed", "employment"),
            ("I was a Product Manager at Microsoft", "job"),
            ("Looking for senior PM roles, preferably remote", "preferences"),
        ]

        print("\nConversation Flow:")
        print("-" * 60)

        robotic_indicators = [
            "thanks for sharing",
            "thank you for sharing",
            "thanks for providing",
            "thank you for providing",
        ]
        
        natural_indicators = [
            "got it",
            "i see",
            "perfect",
            "great",
            "okay",
            "alright",
            "sounds good",
            "excellent",
            "wonderful",
            "i understand",
        ]

        robotic_count = 0
        natural_count = 0

        for i, (msg, context) in enumerate(conversation, 1):
            print(f"\n[Turn {i} - {context}]")
            print(f"User: {msg}")
            
            result = await concierge_graph.ainvoke({"messages": [msg]}, config)
            
            if result.get("direct_response_to_the_user"):
                response = result["direct_response_to_the_user"]
                print(f"AI: {response[:80]}...")
                
                # Check for robotic patterns
                response_lower = response.lower()
                for pattern in robotic_indicators:
                    if pattern in response_lower:
                        robotic_count += 1
                        print(f"  ⚠️  Robotic: '{pattern}'")
                        break
                
                # Check for natural patterns
                for pattern in natural_indicators:
                    if pattern in response_lower:
                        natural_count += 1
                        print(f"  ✅ Natural: Uses '{pattern}'")
                        break
            
            if result.get("selected_agent"):
                print(f"\n✅ Handoff complete!")
                break

        # Final assessment
        print("\n" + "=" * 70)
        print("NATURALNESS ASSESSMENT")
        print("=" * 70)
        
        total_responses = i
        naturalness_score = (natural_count - robotic_count) / total_responses if total_responses > 0 else 0
        
        print(f"Total turns: {total_responses}")
        print(f"Robotic patterns: {robotic_count}")
        print(f"Natural patterns: {natural_count}")
        print(f"Naturalness score: {naturalness_score:.2f}")
        
        if robotic_count == 0:
            print("\n✅ EXCELLENT: No robotic patterns detected!")
        elif robotic_count <= 1:
            print("\n✅ GOOD: Minimal robotic patterns")
        elif robotic_count <= 2:
            print("\n⚠️  ACCEPTABLE: Some robotic patterns present")
        else:
            print("\n❌ POOR: Too many robotic patterns")
        
        if natural_count >= total_responses * 0.7:
            print("✅ Natural language usage is strong")
        elif natural_count >= total_responses * 0.5:
            print("⚠️  Natural language usage could be improved")
        else:
            print("❌ Needs more natural language variety")


async def main() -> None:
    """Run all naturalness tests."""
    print("\n" + "=" * 70)
    print("NATURAL RESPONSE TESTING SUITE")
    print("=" * 70)
    print("\nTesting improvements to conversation naturalness:")
    print("- Reduced 'Thanks for sharing' repetition")
    print("- Increased response variety")
    print("- More natural acknowledgments")

    # Test 1: Response variety
    await test_response_variety()
    
    # Test 2: Overall naturalness
    await test_conversation_naturalness()
    
    print("\n" + "=" * 70)
    print("NATURAL RESPONSE TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
