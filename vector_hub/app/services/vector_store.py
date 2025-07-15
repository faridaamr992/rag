import os
from app.constants.strings import FAISS_INDEX_PATH
from app.services.embedding import embedding_model
from langchain_community.vectorstores import FAISS

if os.path.exists(FAISS_INDEX_PATH):
    faiss_db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    faiss_db = FAISS.from_texts(["dummy"], embedding_model)
    #faiss_db.delete(["0"])
