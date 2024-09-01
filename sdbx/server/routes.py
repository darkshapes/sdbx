from networkx import node_link_graph
from fastapi import APIRouter, WebSocket

from sdbx import config
from sdbx.server.types import Graph

def register_routes(rtr: APIRouter):
    @rtr.get("/nodes")
    async def list_nodes():
        return config.node_manager.node_info
    
    @rtr.post("/prompt")
    async def start_prompt(graph: Graph):
        await config.executor.execute(node_link_graph(graph.dict()))
        return
    
    @rtr.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        await websocket.close()