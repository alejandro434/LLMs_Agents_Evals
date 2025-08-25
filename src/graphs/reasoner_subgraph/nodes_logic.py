"""Reasoner node logic.

uv run -m src.graphs.reasoner_subgraph.nodes_logic

"""

# %%

from typing import Literal

from langgraph.graph import END, MessagesState
from langgraph.types import Command


async def reasoner_node(state: MessagesState) -> Command[Literal[END]]:
    """Reasoner node."""
    print(f"Reasoner node received state: {state}")
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
        response = await reasoner_node(state)
        print(f"Reasoner node response: {response}")

    asyncio.run(main())
