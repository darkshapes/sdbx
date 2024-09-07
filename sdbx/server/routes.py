import json
import uuid
import logging
from asyncio import create_task, wait, FIRST_COMPLETED

from networkx import node_link_graph
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from sdbx import config, logger
from sdbx.server.types import Graph
from sdbx.server.serialize import WebEncoder

def register_routes(rtr: APIRouter):
    @rtr.get("/nodes")
    async def list_nodes():
        return config.node_manager.node_info
    
    @rtr.post("/prompt")
    async def start_prompt(graph: Graph):
        tid = str(uuid.uuid4())
        try:
            task = config.executor.execute(node_link_graph(graph.dict()), tid)
            return {"task_id": tid}
        except Exception as e:
            logger.exception(e)
            return {"error": str(e)}
    
    @rtr.post("/kill/{tid}")
    async def kill_prompt(tid: str):
        try:
            config.executor.halt(tid)
            return {"task_id": tid}
        except Exception as e:
            logger.exception(e)
            return {"error": str(e)}
    
    @rtr.websocket("/ws/{tid}")
    async def websocket_endpoint(websocket: WebSocket, tid: str):
        await websocket.accept()

        task_context = config.executor.tasks.get(tid)

        if not task_context:
            await websocket.send_json({"error": "Invalid task ID"})
            await websocket.close()
            return
        
        websocket.send_serialized = lambda data: websocket.send({
            "type": "websocket.send", 
            "text": json.dumps(data, separators=(",", ":"), ensure_ascii=False, cls=WebEncoder)
        })

        async def notifier():
            try:
                while True:
                    # Wait for either result_event or error_event
                    done, _ = await wait(
                        [
                            create_task(task_context.result_event.wait()), 
                            create_task(task_context.error_event.wait()),
                            create_task(task_context.completion_event.wait())
                        ],
                        return_when=FIRST_COMPLETED
                    )
                    
                    if task_context.error_event.is_set():
                        raise task_context.task_error
                    elif task_context.completion_event.is_set():
                        await websocket.send_serialized({"task_id": tid, "results": dict(task_context.results), "completed": True})
                        await websocket.close()
                        return
                    elif task_context.result_event.is_set():
                        print("sending results")
                        # Send the latest results of the task to the websocket
                        await websocket.send_serialized({"task_id": tid, "results": dict(task_context.results)})

                        # Inform that results are finished processing
                        task_context.process_event.set() # All events are cleared by the executor
            except Exception as e:
                # If error occurred, send error message and close the WebSocket
                logger.exception(e)
                logger.error("sending websocket error")
                await websocket.send_json({"task_id": tid, "error": task_context.task_error})
                await websocket.close()
                return

        ntask = create_task(notifier())

        try:
            # Keep the websocket open
            while True:
                await websocket.receive_text()  # You can use this to receive pings or other messages
        except WebSocketDisconnect:
            # Cleanup when the WebSocket is disconnected
            ntask.cancel()
        finally:
            try:
                await websocket.close()
            except Exception:
                pass