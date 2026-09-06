from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import logging
import time

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel


app = FastAPI(
    title="Clinical AI Cardiomegaly API",
    version="0.1.0",
)

logger = logging.getLogger("clinical_ai_api")
logger.setLevel(logging.INFO)


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
        )


ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
}


class PredictionResponse(BaseModel):
    image_probability: float
    text_probability: float
    fusion_probability: float
    fusion_alpha: float


@lru_cache
def get_predictor():
    # Heavy ML imports and artifact loading happen only
    # when a real prediction is requested.
    from src.srcinference import CardiomegalyPredictor

    return CardiomegalyPredictor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    image: UploadFile = File(...),
    indication: str = Form(...),
    predictor=Depends(get_predictor),
):
    indication = indication.strip()

    if not indication:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Indication cannot be empty.",
        )

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Image must be a PNG or JPEG file.",
        )

    suffix = Path(image.filename or "").suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:
        shutil.copyfileobj(image.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        # Verify that the uploaded bytes are actually an image.
        try:
            with Image.open(temp_path) as uploaded_image:
                uploaded_image.verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid image.",
            )

        try:
            return predictor.predict(
                image_path=temp_path,
                indication=indication,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed.",
            ) from exc

    finally:
        temp_path.unlink(missing_ok=True)