import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from uvicorn.config import LOGGING_CONFIG

import config
from src.api import router as api_router

app = FastAPI()


def main() -> None:
    app.include_router(api_router)

    app.add_middleware(GZipMiddleware, minimum_size=500)
    app.mount("/styles", StaticFiles(directory="web/styles"))
    app.mount("/js", StaticFiles(directory="web/js"))

    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"

    host = os.getenv("SLAE_ATTACK_HOST", "0.0.0.0")
    port = int(os.getenv("SLAE_ATTACK_PORT", config.PORT))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
