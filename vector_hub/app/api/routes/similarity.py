from fastapi import APIRouter, HTTPException
from app.models.schemas import SentenceInput
from app.services.operations import compute_dot_product_similarity

router = APIRouter()

@router.post("/")
async def compute_similarity(data: SentenceInput):
    try:
        return compute_dot_product_similarity(data.sentences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
