from pydantic import BaseModel, Field


class AugmentedQuestions(BaseModel):
    """Augmented question schema."""

    question: str = Field(description="The question to be augmented.")
    augmented_questions: list[str] = Field(description="The augmented questions.")
