from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import TYPE_CHECKING

from fastapi import (
    FastAPI,
)
from fastapi.middleware.cors import CORSMiddleware

from faster_whisper_server.dependencies import get_config, get_model_manager
from faster_whisper_server.logger import setup_logger
from faster_whisper_server.routers.list_models import (
    router as list_models_router,
)
from faster_whisper_server.routers.misc import (
    router as misc_router,
)
from faster_whisper_server.routers.stt import (
    router as stt_router,
)
from fastapi import Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logging.basicConfig(level=logging.DEBUG)




def create_app() -> FastAPI:
    setup_logger()

    logger = logging.getLogger(__name__)

    config = get_config()  # HACK
    logger.debug(f"Config: {config}")

    model_manager = get_model_manager()  # HACK

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        for model_name in config.preload_models:
            model_manager.load_model(model_name)
        yield

    app = FastAPI(lifespan=lifespan)

    app.include_router(stt_router)
    app.include_router(list_models_router)
    app.include_router(misc_router)

    if config.allow_origins is not None:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Set appropriate origin for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(403)
    async def forbidden_error_handler(request: Request, exc):
        print(f">>>>> Error 403 {str(exc)}")
        return JSONResponse(
            status_code=403,
            content={"message": "Forbidden: Check permissions or CORS settings"},
        )

    if config.enable_ui:
        import gradio as gr

        from faster_whisper_server.gradio_app import create_gradio_demo

        app = gr.mount_gradio_app(app, create_gradio_demo(config), path="/")

    return app
