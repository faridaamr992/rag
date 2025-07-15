from fastapi import FastAPI
from app.api.routes import save, search, similarity

app = FastAPI()

app.include_router(save.router, prefix="/save", tags=["Save"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(similarity.router, prefix="/similarity", tags=["Similarity"])
