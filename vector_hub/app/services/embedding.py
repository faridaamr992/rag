from langchain_community.embeddings import HuggingFaceEmbeddings
from app.constants.strings import MINILM_MODEL_NAME

embedding_model = HuggingFaceEmbeddings(model_name=MINILM_MODEL_NAME)
