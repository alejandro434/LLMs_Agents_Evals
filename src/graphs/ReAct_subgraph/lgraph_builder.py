"""Build the ReAct subgraph.

uv run -m src.graphs.ReAct_subgraph.lgraph_builder
"""

# %%
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.graphs.ReAct_subgraph.nodes_logic import react_node, tools_advisor_node
from src.graphs.ReAct_subgraph.schemas import ReActSubgraphState
from src.graphs.receptionist_subgraph.schemas import (
    UserProfileSchema,
    UserRequestExtractionSchema,
)


builder = StateGraph(ReActSubgraphState)

builder.add_node("tools_advisor_node", tools_advisor_node)
builder.add_node("react_node", react_node)

builder.add_edge(START, "tools_advisor_node")

# Explicit edges so Studio shows node connectivity
builder.add_edge("tools_advisor_node", "react_node")
builder.add_edge("react_node", END)

graph_with_in_memory_checkpointer = builder.compile(checkpointer=MemorySaver())
if __name__ == "__main__":
    import asyncio

    async def test_simple_react_graph() -> None:
        """Simple test for ReAct subgraph."""
        # Test with in-memory checkpointer
        print("\n=== Testing ReAct Graph with In-Memory Checkpointer ===")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        test_input = {
            "user_request": UserRequestExtractionSchema(
                task=("Find in the web job fairs events in the next 30 days.")
            ),
            "user_profile": UserProfileSchema(
                name="John Doe",
                current_employment_status="employed",
                zip_code="20850",
                what_is_the_user_looking_for=(
                    "Find job fairs events in the next 30 days"
                ),
            ),
            "why_this_agent_can_help": (
                "I can help the user find job fairs events in the next 30 days."
            ),
        }

        # Run the graph
        result = await graph_with_in_memory_checkpointer.ainvoke(test_input, config)

        # Check results
        print(f"\nSuggested Tools: {result.get('suggested_tools')}")
        print(f"\nTools Advisor Reasoning: {result.get('tools_advisor_reasoning')}")
        final_answer = result.get("final_answer")
        if final_answer:
            print(f"\nFinal Answer: {final_answer}")
        else:
            print("\nFinal Answer not available.")

        assert result.get("suggested_tools") is not None, "Should have suggested tools"
        assert result.get("tools_advisor_reasoning") is not None, (
            "Should have tools advisor reasoning"
        )
        print("\n✅ Test passed!")

    async def test_subgraph_async_streaming() -> None:
        """Test async streaming response with SQLite persistence."""
        print("\n=== Testing ReAct Graph with SQLite Persistence ===")
        async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
            react_graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

            test_input = {
                "user_request": UserRequestExtractionSchema(
                    task=("Find in the web job fairs events in the next 30 days.")
                ),
                "user_profile": UserProfileSchema(
                    name="Jane Smith",
                    current_employment_status="unemployed",
                    zip_code="20852",
                    what_is_the_user_looking_for="Entry-level roles in MD",
                ),
                "why_this_agent_can_help": (
                    "I can help the user find job fairs events in the next 30 days."
                ),
            }

            print("\nStreaming updates:")
            async for update in react_graph.astream(
                test_input,
                config,
                stream_mode="updates",
            ):
                for node_name, node_output in update.items():
                    print(f"  - Node '{node_name}' completed")
                    if "suggested_tools" in node_output:
                        print(f"    Suggested tools: {node_output['suggested_tools']}")
                    if "final_answer" in node_output:
                        print("    Final answer generated")

            # Get final state
            final_state = await react_graph.aget_state(config)
            print("\nFinal state values:")
            print(f"  - Has user_request: {'user_request' in final_state.values}")
            print(
                f"  - Has suggested tools: {final_state.values.get('suggested_tools') is not None}"
            )
            print(
                f"  - Has reasoning: {final_state.values.get('tools_advisor_reasoning') is not None}"
            )
            print("\n✅ Streaming test passed!")

    # Run both tests
    asyncio.run(test_simple_react_graph())
    asyncio.run(test_subgraph_async_streaming())
