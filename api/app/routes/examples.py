from __future__ import annotations

from fastapi import APIRouter

from app import service
from app.schemas import Example

router = APIRouter(prefix="/v1", tags=["examples"])


@router.get(
    "/examples",
    response_model=list[Example],
    summary="Curated pairs with precomputed results (served instantly, no compute)",
)
def list_examples() -> list[Example]:
    return service.examples()
