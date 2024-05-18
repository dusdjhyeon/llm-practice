from fastapi import APIRouter

from app.api.api_v1.endpoints import chat

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/models", tags=["models"])