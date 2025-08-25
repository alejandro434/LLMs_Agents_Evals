"""RAG node logic.

uv run -m src.graphs.rag_subgraph.nodes_logic

"""

# %%
from typing import Literal

from langgraph.graph import END, MessagesState
from langgraph.types import Command


async def rag_node(state: MessagesState) -> Command[Literal[END]]:
    """RAG node."""
    print(f"RAG node received state: {state}")
    return Command(goto=END)


if __name__ == "__main__":
    import asyncio

    async def main():
        """Main function."""
        state = MessagesState(
            messages=[
                {
                    "role": "user",
                    "content": "what is the meaning of life?",
                }
            ]
        )
        response = await rag_node(state)
        print(f"RAG node response: {response}")

    asyncio.run(main())
