from sdbx import config

def create_app():
    from fastapi import FastAPI, APIRouter

    app = FastAPI()
    router = APIRouter()

    from sdbx.server.routes import register_routes

    register_routes(router)
    app.include_router(router)

    from sdbx.server.static import register_static
    register_static(app)

    return app