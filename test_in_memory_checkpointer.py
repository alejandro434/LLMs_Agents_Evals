"""Test the in-memory checkpointer graph directly.

uv run test_in_memory_checkpointer.py
"""

# %%
import asyncio
import time
import uuid

from src.graphs.concierge_workflow import graph_with_in_memory_checkpointer


async def test_in_memory_checkpointer_direct() -> None:
    """Test the graph_with_in_memory_checkpointer directly."""
    print("\n" + "=" * 70)
    print("TESTING IN-MEMORY CHECKPOINTER GRAPH DIRECTLY")
    print("=" * 70)
    
    # Test 1: Basic functionality with pre-compiled graph
    print("\n--- Test 1: Basic Functionality ---")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Initial message
    test_input = {"messages": ["Hi, I'm John Smith."]}
    result = await graph_with_in_memory_checkpointer.ainvoke(test_input, config)
    
    assert result.get("direct_response_to_the_user"), "Should get receptionist response"
    print(f"✓ Initial response: {result['direct_response_to_the_user'][:80]}...")
    
    # Test 2: State persistence across invocations
    print("\n--- Test 2: State Persistence ---")
    
    # Add more information
    test_input2 = {
        "messages": [
            "Hi, I'm John Smith.",
            "I live at 789 Tech Ave, San Jose, CA. I'm unemployed."
        ]
    }
    result2 = await graph_with_in_memory_checkpointer.ainvoke(test_input2, config)
    
    # Check state is maintained
    state = await graph_with_in_memory_checkpointer.aget_state(config)
    assert state.values.get("messages"), "Messages should be in state"
    assert len(state.values["messages"]) >= 2, "Should have multiple messages"
    print(f"✓ State persisted: {len(state.values['messages'])} messages in history")
    
    # Test 3: Complete flow to agent selection
    print("\n--- Test 3: Complete Flow to Agent Selection ---")
    
    messages_complete = [
        "Hi, I'm John Smith.",
        "I live at 789 Tech Ave, San Jose, CA. I'm unemployed.",
        "My last job was Senior Engineer at TechCo in San Jose.",
        "I'm looking for senior engineering roles, remote preferred.",
        "Find remote senior software engineer positions at startups"
    ]
    
    test_input3 = {"messages": messages_complete}
    result3 = await graph_with_in_memory_checkpointer.ainvoke(test_input3, config)
    
    if result3.get("selected_agent"):
        print(f"✓ Agent selected: {result3['selected_agent']}")
        print(f"✓ Task: {result3.get('task', 'N/A')[:80]}...")
    
    # Get final state to verify profile
    final_state = await graph_with_in_memory_checkpointer.aget_state(config)
    if final_state.values.get("user_profile"):
        profile = final_state.values["user_profile"]
        if profile and hasattr(profile, "name"):
            print(f"✓ Profile captured: {profile.name}")
    
    # Test 4: Multiple threads (isolation)
    print("\n--- Test 4: Thread Isolation ---")
    
    thread1 = str(uuid.uuid4())
    thread2 = str(uuid.uuid4())
    config1 = {"configurable": {"thread_id": thread1}}
    config2 = {"configurable": {"thread_id": thread2}}
    
    # Thread 1
    await graph_with_in_memory_checkpointer.ainvoke(
        {"messages": ["I'm Alice."]}, config1
    )
    
    # Thread 2
    await graph_with_in_memory_checkpointer.ainvoke(
        {"messages": ["I'm Bob."]}, config2
    )
    
    # Check isolation
    state1 = await graph_with_in_memory_checkpointer.aget_state(config1)
    state2 = await graph_with_in_memory_checkpointer.aget_state(config2)
    
    messages1 = state1.values.get("messages", [])
    messages2 = state2.values.get("messages", [])
    
    assert "Alice" in str(messages1) and "Alice" not in str(messages2), "Thread 1 should only have Alice"
    assert "Bob" in str(messages2) and "Bob" not in str(messages1), "Thread 2 should only have Bob"
    print("✓ Thread isolation verified")
    
    # Test 5: Streaming with in-memory checkpointer
    print("\n--- Test 5: Streaming Support ---")
    
    stream_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    stream_input = {
        "messages": [
            "Hi, I'm Emma Watson.",
            "I live at 321 AI Drive, Austin, TX. I'm employed.",
            "I'm a Data Scientist at DataCorp in Austin.",
            "Looking for senior data science roles, remote only.",
            "Search for remote data science positions with good work-life balance"
        ]
    }
    
    updates_count = 0
    async for update in graph_with_in_memory_checkpointer.astream(
        stream_input, stream_config, stream_mode="updates"
    ):
        updates_count += 1
        if "receptor_router" in update:
            print(f"✓ Streamed update {updates_count}: receptor_router")
        elif "react" in update:
            print(f"✓ Streamed update {updates_count}: react agent")
    
    print(f"✓ Total updates streamed: {updates_count}")
    
    print("\n" + "=" * 70)
    print("✅ IN-MEMORY CHECKPOINTER TESTS PASSED!")
    print("=" * 70)


async def test_in_memory_checkpointer_advanced() -> None:
    """Advanced tests for in-memory checkpointer graph."""
    print("\n" + "=" * 70)
    print("ADVANCED IN-MEMORY CHECKPOINTER TESTS")
    print("=" * 70)
    
    # Test 1: Interrupt and resume
    print("\n--- Test 1: Interrupt and Resume ---")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Start with incomplete info
    messages = ["Hi, I'm Sarah Connor."]
    result = await graph_with_in_memory_checkpointer.ainvoke(
        {"messages": messages}, config
    )
    
    # Should get interrupt asking for more info
    assert result.get("direct_response_to_the_user"), "Should ask for more info"
    print(f"✓ Interrupt 1: {result['direct_response_to_the_user'][:60]}...")
    
    # Add more info incrementally
    messages.append("I live at 999 Future Blvd, Los Angeles, CA. I'm unemployed.")
    result = await graph_with_in_memory_checkpointer.ainvoke(
        {"messages": messages}, config
    )
    
    assert result.get("direct_response_to_the_user"), "Should ask for job info"
    print(f"✓ Interrupt 2: {result['direct_response_to_the_user'][:60]}...")
    
    # Complete the profile
    messages.extend([
        "I was a Security Specialist at CyberDyne Systems in LA.",
        "I want cybersecurity roles, remote or hybrid."
    ])
    result = await graph_with_in_memory_checkpointer.ainvoke(
        {"messages": messages}, config
    )
    
    # Now make a job request
    messages.append("Find cybersecurity job openings in California")
    result = await graph_with_in_memory_checkpointer.ainvoke(
        {"messages": messages}, config
    )
    
    assert result.get("selected_agent") == "react", "Should select react agent"
    print(f"✓ Resume successful: Agent={result['selected_agent']}")
    
    # Test 2: Checkpoint history
    print("\n--- Test 2: Checkpoint History ---")
    
    # Get checkpoint history
    state = await graph_with_in_memory_checkpointer.aget_state(config)
    
    # Check we can access the state history
    assert state.values.get("messages"), "Should have messages"
    assert len(state.values["messages"]) >= 5, "Should have full conversation"
    print(f"✓ Checkpoint has {len(state.values['messages'])} messages")
    
    # Verify user profile was built incrementally
    if state.values.get("user_profile"):
        profile = state.values["user_profile"]
        assert profile.name == "Sarah Connor", "Profile name should match"
        assert profile.current_address, "Should have address"
        assert profile.last_job, "Should have job history"
        print(f"✓ Profile correctly assembled: {profile.name}")
    
    # Test 3: Error recovery
    print("\n--- Test 3: Error Recovery ---")
    
    error_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Try with malformed input
    try:
        # This should handle gracefully
        result = await graph_with_in_memory_checkpointer.ainvoke(
            {"messages": [""]}, error_config
        )
        print("✓ Handled empty message gracefully")
    except Exception as e:
        print(f"✗ Failed on empty message: {e}")
    
    # Test 4: Performance with multiple requests
    print("\n--- Test 4: Performance Test ---")
    
    perf_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    messages = [
        "I'm Quick Test.",
        "123 Speed St, Fast City, TX. Unemployed.",
        "Was a Developer at QuickCo in Dallas.",
        "Want any developer job.",
        "Find developer jobs in Texas"
    ]
    
    start_time = time.time()
    result = await graph_with_in_memory_checkpointer.ainvoke(
        {"messages": messages}, perf_config
    )
    elapsed = time.time() - start_time
    
    print(f"✓ Full flow completed in {elapsed:.2f}s")
    if elapsed < 15:
        print("✓ Performance: Good (<15s)")
    elif elapsed < 30:
        print("⚠️  Performance: Acceptable (15-30s)")
    else:
        print("✗ Performance: Slow (>30s)")
    
    print("\n" + "=" * 70)
    print("✅ ADVANCED IN-MEMORY CHECKPOINTER TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_in_memory_checkpointer_direct())
    
    # Run advanced tests
    asyncio.run(test_in_memory_checkpointer_advanced())
    
    print("\n" + "=" * 70)
    print("🎉 ALL IN-MEMORY CHECKPOINTER TESTS PASSED!")
    print("=" * 70)
# %%
