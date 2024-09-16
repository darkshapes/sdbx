from sdbx import config

def create_app():
    from fastapi import FastAPI, APIRouter
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()
    router = APIRouter()

    from sdbx.server.routes import register_routes
    
    register_routes(router)
    app.include_router(router)

    # Mount the built React app's static files
    app.mount("/", StaticFiles(directory=config.client_manager.selected_path, html=True), name="static")

    return app