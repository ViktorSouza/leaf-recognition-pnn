from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from constants import SERVICE_NAME
from middlewares.internal_token import InternalTokenMiddleware
from models.leaf_pnn import LeafPNN
from repositories.leaf_model_repository import load_classifier
from schemas.prediction import PredictionResponse
from sources.leaf_feature_extractor import InvalidLeafImage, extract_leaf_features
from utils.logger import setup_logging

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    application.state.classifier = load_classifier()
    yield


def create_app() -> FastAPI:
    """Monta a aplicação FastAPI com o classificador de folhas."""
    application = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(InternalTokenMiddleware)

    @application.get("/")
    def root() -> dict[str, str]:
        return {"service": SERVICE_NAME, "status": "ok"}

    @application.get("/health_check")
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/predict", response_model=PredictionResponse)
    def predict_leaf(
        request: Request,
        image: UploadFile = File(...),
    ) -> PredictionResponse:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Envie um arquivo de imagem.")

        image_bytes = image.file.read(MAX_IMAGE_SIZE_BYTES + 1)
        if not image_bytes:
            raise HTTPException(status_code=422, detail="O arquivo enviado está vazio.")
        if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="A imagem excede o limite de 10 MB.",
            )

        try:
            features = extract_leaf_features(image_bytes)
        except InvalidLeafImage as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        classifier: LeafPNN = request.app.state.classifier
        try:
            result = classifier.predict(features)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return PredictionResponse(
            label=result.label,
            species=result.species,
            confidence=result.confidence,
        )

    return application


def main() -> FastAPI:
    setup_logging()
    return create_app()


app = main()
