from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List
import numpy as np
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from strings import MINILM_MODEL_NAME

app = FastAPI()

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name=MINILM_MODEL_NAME)

# Path to save/load FAISS index
FAISS_INDEX_PATH = "./faiss_index"

# Load FAISS if it exists
faiss_db = None
if os.path.exists(FAISS_INDEX_PATH):
    try:
        faiss_db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model)
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        faiss_db = None


# ---------- Input Schemas ----------

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


# ---------- Endpoints ----------

@app.post("/save/")
async def save_sentences(data: SentenceInput):
    global faiss_db
    try:
        if faiss_db is None:
            faiss_db = FAISS.from_texts(data.sentences, embedding_model)
        else:
            faiss_db.add_texts(data.sentences)

        faiss_db.save_local(FAISS_INDEX_PATH)
        return {"message": "Sentences saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/")
async def search_similar(data: SearchInput):
    try:
        if faiss_db is None:
            raise HTTPException(status_code=404, detail="No sentences in FAISS database yet.")

        results = faiss_db.similarity_search(data.query, k=data.top_k)
        return {
            "query": data.query,
            "results": [r.page_content for r in results]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity/")
async def compute_similarity(data: SentenceInput):
    try:
        embeddings = embedding_model.embed_documents(data.sentences)
        vec1, vec2, vec3 = embeddings

        dot_12 = np.dot(vec1, vec2)
        dot_13 = np.dot(vec1, vec3)
        dot_23 = np.dot(vec2, vec3)

        similarities = {
            "sentence_1_vs_2": dot_12,
            "sentence_1_vs_3": dot_13,
            "sentence_2_vs_3": dot_23
        }

        most_similar = max(similarities, key=similarities.get)

        return {
            "dot_products": similarities,
            "most_similar_pair": most_similar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
