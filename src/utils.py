"""Utility functions for the agentic workflow.

uv run python src/utils.py
"""

# %%
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv(override=True)


def get_llm():
    """Get the LLM chain."""
    return ChatOpenAI(
        model="gpt-4.1-nano",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        """This is a test function to test the LLM."""
        result = await get_llm().ainvoke("What is the capital of France?")
        print(result)
        return result

    asyncio.run(main())

# %%
