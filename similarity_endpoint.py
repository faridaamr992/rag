from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from strings import MINILM_MODEL_NAME
from typing import List

app = FastAPI()

# Load model once when the app starts
embedding_model = HuggingFaceEmbeddings(model_name=MINILM_MODEL_NAME)

# Request body schema
class SentenceInput(BaseModel):
    sentences: List[str]

    @validator("sentences")
    def check_exactly_three_sentences(cls, v):
        if len(v) != 3:
            raise ValueError("Exactly 3 sentences are required.")
        return v

@app.post("/similarity/")
async def compute_similarity(data: SentenceInput):
    try:
        # Generate embeddings
        embeddings = embedding_model.embed_documents(data.sentences)
        vec1, vec2, vec3 = embeddings

        # Compute dot products
        dot_12 = np.dot(vec1, vec2)
        dot_13 = np.dot(vec1, vec3)
        dot_23 = np.dot(vec2, vec3)

        # Store dot products with their pairs
        similarities = {
            "sentence_1_vs_2": dot_12,
            "sentence_1_vs_3": dot_13,
            "sentence_2_vs_3": dot_23
        }

        # Determine most similar pair
        most_similar = max(similarities, key=similarities.get)

        return {
            "dot_products": similarities,
            "most_similar_pair": most_similar
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
