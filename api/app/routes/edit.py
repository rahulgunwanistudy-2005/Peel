from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from peel.io import decode_image, encode_png
from starlette.concurrency import run_in_threadpool

from app import gemini
from app.limits import rate_limit, read_upload
from app.schemas import ErrorBody

router = APIRouter(prefix="/v1", tags=["edit"])


@router.post(
    "/edit",
    summary="Edit an image with Nano Banana (Gemini), server-side",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "The edited image. `X-Peel-Edit-Model` names the model.",
        },
        503: {
            "model": ErrorBody,
            "description": "EDIT_UNAVAILABLE: no key, no quota, or no image model.",
        },
    },
    dependencies=[Depends(rate_limit(3.0))],
)
async def edit(
    original: Annotated[UploadFile, File(description="The image to edit.")],
    instruction: Annotated[str, Form(max_length=2000)],
) -> Response:
    data = await read_upload(original, "original image")
    # Re-encode what we decoded, so the model sees exactly the pixels Peel will compare against.
    decoded = decode_image(data, max_long_side=2048, label="original image")
    png = encode_png(decoded.rgb)
    image, model = await run_in_threadpool(gemini.edit_image, png, "image/png", instruction.strip())
    out = encode_png(decode_image(image, label="edited image").rgb)
    return Response(content=out, media_type="image/png", headers={"X-Peel-Edit-Model": model})
