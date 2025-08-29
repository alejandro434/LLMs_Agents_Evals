"""Websearch tool.

uv run -m src.graphs.ReAct_subgraph.tools.websearch

"""

# %%

from dotenv import load_dotenv
from langchain_tavily import TavilySearch


load_dotenv(override=True)

websearch_tool = TavilySearch(
    max_results=5,
    topic="general",
    include_answer=False,
    include_raw_content=False,
    include_images=False,
    include_image_descriptions=False,
    search_depth="basic",
    time_range="day",
    include_domains=None,
    exclude_domains=None,
)

if __name__ == "__main__":
    result = websearch_tool.invoke("What happened at the last wimbledon?")
    print(result)
