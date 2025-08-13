import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

from sdbx import config


def register_static(app: FastAPI):
    @app.get("/{path:path}")
    async def static(path: str):
        if os.path.exists(file := os.path.join(config.client_manager.selected_path, path or "index.html")):
            return FileResponse(file)
        else:
            return PlainTextResponse("404: Not Found", status_code=404)
