from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles

from sdbx import config
from sdbx.server.routes import register_routes

def create_app():
    api_router = APIRouter()
    register_routes(api_router)

    app = FastAPI()
    app.include_router(api_router)

    # Mount the built React app's static files
    app.mount("/", StaticFiles(directory=config.client_manager.selected_path, html=True), name="static")

    return app