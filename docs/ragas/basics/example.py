"""Example of using Ragas to evaluate a model's performance.

uv run src/ragas/example.py
"""

# %%
from ragas import SingleTurnSample
from ragas.metrics import BleuScore


test_data = {
    "user_input": (
        "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by"
        "strong performance in the Asian market. Sales in this region have significantly"
        "contributed to the overall growth. Analysts attribute this success to strategic"
        "marketing and product localization. The positive trend in the Asian market is"
        "expected to continue into the next quarter."
    ),
    "response": (
        "The company experienced an 8% increase in Q3 2024, largely due to effective"
        "marketing strategies and product adaptation, with expectations of continued"
        "growth in the coming quarter."
    ),
    "reference": (
        "The company reported an 8% growth in Q3 2024, primarily driven by strong sales"
        "in the Asian market, attributed to strategic marketing and localized products,"
        "with continued growth anticipated in the next quarter."
    ),
}
if __name__ == "__main__":
    metric = BleuScore()
    test_data = SingleTurnSample(**test_data)
    metric.single_turn_score(test_data)

# %%
