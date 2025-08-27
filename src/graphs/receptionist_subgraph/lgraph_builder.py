"""Build the receptionist subgraph.

uv run -m src.graphs.receptionist_subgraph.lgraph_builder
"""


# %%

import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.graphs.receptionist_subgraph.nodes_logic import (
    handoff_to_logging,
    receptor,
    validate_user_profile,
)
from src.graphs.receptionist_subgraph.schemas import ReceptionistSubgraphState


builder = StateGraph(ReceptionistSubgraphState)

builder.add_node("receptor", receptor)
builder.add_node("validate_user_profile", validate_user_profile)
builder.add_node("handoff_to_logging", handoff_to_logging)
builder.add_edge(START, "receptor")

# Explicit edges so Studio shows node connectivity
builder.add_edge("receptor", "validate_user_profile")
builder.add_edge("validate_user_profile", "receptor")
builder.add_edge("validate_user_profile", "handoff_to_logging")
builder.add_edge("handoff_to_logging", END)

# Compile a server-exportable subgraph with persistent checkpointing
sqlite_saver = SqliteSaver.from_conn_string("checkpoints.sqlite")
subgraph = builder.compile(checkpointer=sqlite_saver)


if __name__ == "__main__":
    import asyncio

    async def test_subgraph_async_streaming() -> None:
        """Test async streaming response with in-memory async SQLite persistence."""
        async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
            receptionist_graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": uuid.uuid4()}}
            async for _ in receptionist_graph.astream(
                {"messages": ["Hi, I'm looking for an entry-level retail job."]},
                config,
                stream_mode="updates",
                debug=True,
            ):
                pass

    asyncio.run(test_subgraph_async_streaming())

    async def test_subgraph_sync_invoke() -> None:
        """Test sync invoke of the subgraph."""
        async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
            receptionist_graph = builder.compile(checkpointer=saver)
            config = {"configurable": {"thread_id": uuid.uuid4()}}
            response = await receptionist_graph.ainvoke(
                {"messages": ["Hi, I'm looking for an entry-level retail job."]},
                config,
            )
        print(response)

    asyncio.run(test_subgraph_sync_invoke())
