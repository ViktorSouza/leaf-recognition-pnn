from fastapi import FastAPI
from utils.logger import setup_logging

app = FastAPI()


def create_app() -> FastAPI:
    """Monta a aplicação FastAPI com middlewares, rotas e tratadores de erro."""
    application = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    return application


def main() -> FastAPI:
    setup_logging()
    return create_app()


app = main()
