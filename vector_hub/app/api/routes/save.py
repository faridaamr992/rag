from fastapi import APIRouter, HTTPException
from app.models.schemas import SentenceInput
from app.services.operations import save_sentences_to_faiss

router = APIRouter()

@router.post("/")
async def save_sentences(data: SentenceInput):
    try:
        return save_sentences_to_faiss(data.sentences)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
