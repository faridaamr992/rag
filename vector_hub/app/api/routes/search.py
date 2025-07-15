from fastapi import APIRouter, HTTPException
from app.models.schemas import SearchInput
from app.services.operations import search_similar_sentences

router = APIRouter()

@router.post("/")
async def search_similar(data: SearchInput):
    try:
        return {
            "query": data.query,
            "results": search_similar_sentences(data.query, data.top_k)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
