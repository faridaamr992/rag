import numpy as np
from app.repositories.faiss_repository import add_sentences, search_similar
from app.services.embedding import embedding_model

def save_sentences_to_faiss(sentences):
    add_sentences(sentences)
    return {"message": "Sentences saved successfully."}

def search_similar_sentences(query, top_k):
    results = search_similar(query, k=top_k)
    return [r.page_content for r in results]

def compute_dot_product_similarity(sentences):
    embeddings = embedding_model.embed_documents(sentences)
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
