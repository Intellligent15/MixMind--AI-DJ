from fastapi import APIRouter, Depends, Query

from app.api.deps import get_youtube_service
from app.schemas import SearchResultSchema
from app.services.youtube import YouTubeService

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search", response_model=list[SearchResultSchema])
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=20),
    yt: YouTubeService = Depends(get_youtube_service),
) -> list[SearchResultSchema]:
    results = yt.search(q, limit=limit)
    return [SearchResultSchema(**r.__dict__) for r in results]
