from pydantic import BaseModel, validator
from typing import List

class SentenceInput(BaseModel):
    sentences: List[str]

    @validator("sentences")
    def check_exactly_three_sentences(cls, v):
        if len(v) != 3:
            raise ValueError("Exactly 3 sentences are required.")
        return v

class SearchInput(BaseModel):
    query: str
    top_k: int = 1
