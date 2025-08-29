"""Test suite for user_request_extraction and agent_selection chains.

uv run -m src.graphs.receptionist_subgraph.test_new_chains
"""

# %%
import asyncio

from src.graphs.receptionist_subgraph.chains import (
    AgentSelectionSchema,
    UserRequestExtractionSchema,
    get_agent_selection_chain,
    get_user_request_extraction_chain,
)


async def test_user_request_extraction_basic():
    """Test basic functionality of user request extraction chain."""
    chain = get_user_request_extraction_chain()

    test_cases = [
        {
            "input": "I need to find job fairs happening in my area next month",
            "expected_keywords": ["job fairs", "area", "month"],
        },
        {
            "input": "Can you help me look for remote software engineering positions?",
            "expected_keywords": ["remote", "software engineering"],
        },
        {
            "input": "Help me find entry-level retail jobs near Arlington, VA",
            "expected_keywords": ["entry-level", "retail", "Arlington"],
        },
    ]

    for test_case in test_cases:
        result = await chain.ainvoke(test_case["input"])
        assert isinstance(result, UserRequestExtractionSchema)
        assert result.task, "Task should not be empty"

        # Check that key concepts are preserved in the extracted task
        task_lower = result.task.lower()
        for keyword in test_case["expected_keywords"]:
            assert keyword.lower() in task_lower or any(
                word in task_lower for word in keyword.lower().split()
            ), f"Expected '{keyword}' to be in extracted task: {result.task}"


async def test_agent_selection_basic():
    """Test basic functionality of agent selection chain."""
    chain = get_agent_selection_chain()

    test_tasks = [
        "Search for job fairs in the user's area",
        "Find remote software engineering positions",
        "Look up training programs for healthcare workers",
    ]

    for task in test_tasks:
        result = await chain.ainvoke(task)
        assert isinstance(result, AgentSelectionSchema)
        assert result.agent_name == "react", "Should always select 'react' agent"
        assert result.rationale_of_the_handoff, "Rationale should not be empty"
        assert len(result.rationale_of_the_handoff) > 20, (
            "Rationale should be descriptive"
        )


async def test_extraction_edge_cases():
    """Test edge cases for user request extraction."""
    chain = get_user_request_extraction_chain()

    edge_cases = [
        # Very short input
        "jobs",
        # Long, complex input
        (
            "I'm a recent college graduate with a degree in computer science and I'm "
            "looking for entry-level software development positions, preferably in "
            "startups or mid-size companies that offer good learning opportunities "
            "and mentorship programs, ideally with a focus on web development using "
            "modern JavaScript frameworks"
        ),
        # Question format
        "What kind of certifications do I need for cybersecurity?",
        # Multiple requests
        "Find me nursing jobs and also information about getting my RN license",
    ]

    for input_text in edge_cases:
        result = await chain.ainvoke(input_text)
        assert result.task, (
            f"Should extract task even from edge case: {input_text[:50]}..."
        )
        assert len(result.task) > 5, "Task should be meaningful"


async def test_agent_selection_consistency():
    """Test that agent selection is consistent for similar tasks."""
    chain = get_agent_selection_chain()

    similar_tasks = [
        "Find job openings in Baltimore",
        "Search for job opportunities in Baltimore",
        "Look for available positions in Baltimore",
    ]

    results = []
    for task in similar_tasks:
        result = await chain.ainvoke(task)
        results.append(result)

    # All should select the same agent
    assert all(r.agent_name == "react" for r in results)
    # All should have rationales
    assert all(r.rationale_of_the_handoff for r in results)


async def test_integration_pipeline():
    """Test the full pipeline from user input to agent selection."""
    extraction_chain = get_user_request_extraction_chain()
    selection_chain = get_agent_selection_chain()

    test_scenarios = [
        {
            "user_input": "I need help finding IT certification programs",
            "expected_task_keywords": ["IT", "certification", "programs"],
            "expected_agent": "react",
        },
        {
            "user_input": "Show me warehouse jobs that have benefits",
            "expected_task_keywords": ["warehouse", "jobs", "benefits"],
            "expected_agent": "react",
        },
        {
            "user_input": "Can you find career fairs happening this weekend?",
            "expected_task_keywords": ["career fairs", "weekend"],
            "expected_agent": "react",
        },
    ]

    for scenario in test_scenarios:
        # Extract task
        extraction_result = await extraction_chain.ainvoke(scenario["user_input"])
        assert extraction_result.task

        # Check task contains expected concepts (more flexible matching)
        task_lower = extraction_result.task.lower()
        for keyword in scenario["expected_task_keywords"]:
            # Check if the keyword or any of its words appear in the task
            keyword_lower = keyword.lower()
            keyword_found = (
                keyword_lower in task_lower
                or any(word in task_lower for word in keyword_lower.split())
                or
                # Also check for related terms (e.g., "jobs" -> "job", "openings", "positions")
                (
                    keyword_lower == "jobs"
                    and any(
                        term in task_lower
                        for term in ["job", "position", "opening", "opportunit"]
                    )
                )
            )
            assert keyword_found, (
                f"Expected '{keyword}' or related term in task: {extraction_result.task}"
            )

        # Select agent
        selection_result = await selection_chain.ainvoke(extraction_result.task)
        assert selection_result.agent_name == scenario["expected_agent"]
        assert selection_result.rationale_of_the_handoff


async def test_chain_parameters():
    """Test that chains work with different parameters."""
    # Test with different k values
    for k in [0, 3, 10]:
        chain = get_user_request_extraction_chain(k=k)
        result = await chain.ainvoke("Find me a job")
        assert result.task

    # Test with different temperatures
    for temp in [0, 0.5, 1.0]:
        chain = get_agent_selection_chain(temperature=temp)
        result = await chain.ainvoke("Search for jobs in my area")
        assert result.agent_name == "react"


async def main():
    """Run all tests."""
    print("Running User Request Extraction and Agent Selection Chain Tests")
    print("=" * 70)

    tests = [
        ("Basic User Request Extraction", test_user_request_extraction_basic),
        ("Basic Agent Selection", test_agent_selection_basic),
        ("Extraction Edge Cases", test_extraction_edge_cases),
        ("Agent Selection Consistency", test_agent_selection_consistency),
        ("Integration Pipeline", test_integration_pipeline),
        ("Chain Parameters", test_chain_parameters),
    ]

    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            await test_func()
            print(f"   ✅ {test_name} passed")
        except AssertionError as e:
            print(f"   ❌ {test_name} failed: {e}")
        except Exception as e:
            print(f"   ⚠️ {test_name} error: {e}")

    print("\n" + "=" * 70)
    print("✨ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
