"""Build the Qualifier subgraph.

uv run -m src.graphs.qualifier.lgraph_builder
"""

# %%

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from src.graphs.qualifier.nodes_logic import collect_user_info, qualify_user
from src.graphs.qualifier.schemas import QualifierSubgraphState


builder = StateGraph(QualifierSubgraphState)

builder.add_node("collect_user_info", collect_user_info)
builder.add_node("qualify_user", qualify_user)

builder.add_edge(START, "collect_user_info")


subgraph = builder.compile()
graph_with_in_memory_checkpointer = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    import asyncio

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    async def test_subgraph() -> None:
        """Test the subgraph."""
        input_1 = {"messages": ["Hi, I'm John Smith."]}
        input_2 = {"messages": ["I'm 17 years old."]}
        input_3 = {"messages": ["My zip code is 20850."]}

        response = await graph_with_in_memory_checkpointer.ainvoke(
            input_1, config, debug=True
        )
        response = await graph_with_in_memory_checkpointer.ainvoke(
            input_2, config, debug=True
        )
        response = await graph_with_in_memory_checkpointer.ainvoke(
            input_3, config, debug=True
        )
        print(f"Qualifier response: {response}")

    asyncio.run(test_subgraph())
