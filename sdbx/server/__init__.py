from sdbx import config

def create_app():
    import os
    
    from fastapi import FastAPI, APIRouter
    from fastapi.responses import FileResponse

    app = FastAPI()
    router = APIRouter()

    from sdbx.server.routes import register_routes
    
    register_routes(router)
    app.include_router(router)

    @app.get("/{path:path}")
    async def static(path: str):
        return FileResponse(os.path.join(config.client_manager.selected_path, path or "index.html"))

    return app