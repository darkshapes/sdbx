import uuid
from asyncio import create_task

from networkx import node_link_graph
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from sdbx import config
from sdbx.server.types import Graph

def register_routes(rtr: APIRouter):
    @rtr.get("/nodes")
    async def list_nodes():
        return config.node_manager.node_info
    
    @rtr.post("/prompt")
    async def start_prompt(graph: Graph):
        tid = str(uuid.uuid4())
        task = config.executor.execute(node_link_graph(graph.dict()), tid)
        return PlainTextResponse(tid)
    
    @rtr.post("/kill/{tid}")
    async def kill_prompt(tid: str):
        config.executor.halt(tid)
        return PlainTextResponse(tid)
    
    @rtr.websocket("/ws/{tid}")
    async def websocket_endpoint(websocket: WebSocket, tid: str):
        await websocket.accept()

        task_context = config.executor.tasks.get(tid)

        if not task_context:
            await websocket.send_json({"error": "Invalid task ID"})
            await websocket.close()
            return

        async def notifier():
            while True:
                # Wait for a new result to be available for this specific task
                await task_context.result_event.wait()
                
                # Send the latest results of the task to the websocket
                await websocket.send_json({"task_id": tid, "results": dict(task_context.results)})
                
                # Clear the event to wait for the next result
                task_context.result_event.clear()

        ntask = create_task(notifier())

        try:
            # Keep the websocket open
            while True:
                await websocket.receive_text()  # You can use this to receive pings or other messages
        except WebSocketDisconnect:
            # Cleanup when the WebSocket is disconnected
            ntask.cancel()
        finally:
            await websocket.close()