from app.services.vector_store import faiss_db, FAISS_INDEX_PATH
from typing import List

def add_sentences(sentences: List[str]):
    faiss_db.add_texts(sentences)
    faiss_db.save_local(FAISS_INDEX_PATH)

def search_similar(query: str, k: int):
    return faiss_db.similarity_search(query, k=k)
