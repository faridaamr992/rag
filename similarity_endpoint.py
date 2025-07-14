from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

app = FastAPI()

# Load model once when the app starts
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Request body schema
class SentenceInput(BaseModel):
    sentences: List[str]

@app.post("/similarity/")
async def compute_similarity(data: SentenceInput):
    try:
        if len(data.sentences) != 3:
            raise HTTPException(status_code=400, detail="Exactly 3 sentences are required.")

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
