"""ReAct node logic.

uv run -m src.graphs.ReAct_subgraph.nodes_logic

"""

# %%

from typing import Literal

from langgraph.graph import END, MessagesState
from langgraph.types import Command


async def react_node(state: MessagesState) -> Command[Literal[END]]:
    """ReAct node."""
    print(f"ReAct node received state: {state}")
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
        response = await react_node(state)
        print(f"ReAct node response: {response}")

    asyncio.run(main())
