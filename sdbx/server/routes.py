import json
import logging
import os
import uuid
from asyncio import FIRST_COMPLETED, create_task, wait
from websockets.asyncio.server import broadcast

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from networkx import node_link_graph

from sdbx import config, logger
from sdbx.server.serialize import WebEncoder
from sdbx.server.types import Graph, TaskUpdate


def register_update_signal(rtr: APIRouter):
    @rtr.websocket("/ws/update")
    async def update_signal_websocket(websocket: WebSocket):
        await websocket.accept()


def register_node_routes(rtr: APIRouter):
    @rtr.get("/nodes")
    def list_nodes():
        return config.node_manager.node_info

    @rtr.post("/prompt")  # hate that this is async for reasons of conformity
    async def start_prompt(graph: Graph):
        tid = str(uuid.uuid4())
        try:
            config.executor.execute(node_link_graph(graph.dict()), tid)
            return TaskUpdate(id=tid).dict()
        except Exception as e:
            logger.exception(e)
            return TaskUpdate(id=tid, error=str(e)).dict()

    @rtr.post("/kill/{tid}")
    def kill_prompt(tid: str):
        try:
            config.executor.halt(tid)
            return TaskUpdate(id=tid).dict()
        except Exception as e:
            logger.exception(e)
            return TaskUpdate(id=tid, error=str(e)).dict()

    @rtr.websocket("/ws/task/{tid}")
    async def task_subscribe_websocket(websocket: WebSocket, tid: str):
        await websocket.accept()

        task_context = config.executor.tasks.get(tid)

        if not task_context:
            await websocket.send_json(TaskUpdate(error="Invalid task ID").dict())
            await websocket.close()
            return

        websocket.send_update = lambda **kwargs: websocket.send({"type": "websocket.send", "text": json.dumps(TaskUpdate(id=tid, **kwargs).dict(), separators=(",", ":"), ensure_ascii=False, cls=WebEncoder)})

        async def notifier():
            try:
                while True:
                    # Wait for either result_event or error_event
                    await wait([create_task(task_context.result_event.wait()), create_task(task_context.error_event.wait()), create_task(task_context.completion_event.wait())], return_when=FIRST_COMPLETED)

                    if task_context.error_event.is_set():
                        raise task_context.task_error
                    elif task_context.completion_event.is_set():
                        await websocket.send_update(results=dict(task_context.results), completed=True)
                        await websocket.close()
                        return
                    elif task_context.result_event.is_set():
                        print("sending results")
                        # Send the latest results of the task to the websocket
                        await websocket.send_update(results=dict(task_context.results))

                        # Inform that results are finished processing
                        task_context.process_event.set()  # All events are cleared by the executor
            except Exception as e:
                # If error occurred, send error message and close the WebSocket
                logger.exception(e)
                logger.error("sending websocket error")
                await websocket.send_update(error=str(task_context.task_error))
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

    @rtr.post("/tune/{node_id}")
    def tune_node(node_id: str, graph: Graph):
        try:
            g = node_link_graph(graph.model_dump())
            node_fn = config.node_manager.registry[g.nodes[node_id]["fname"]]
            params = node_fn.tuner.collect_tuned_parameters(config.node_manager, g, node_id)
            return {"tuned_parameters": params}
        except Exception as e:
            logger.exception(e)
            return {"error": str(e)}

    # @rtr.post("/add/{node_id}")
    # async def add_nodes(node_name: str, graph: Graph):
    #     from sdbx.server.types import Node,
    #     from sdbx.nodes.base import nodes as base_nodes
    #     from importlib import import_module

    #     try:
    #         g = node_link_graph(graph.model_dump())
    #         node_data = import_module(
    #             base_nodes,
    #             node_name,
    #         )
    #         # node_data = {
    #         #     "id": str(uuid.uuid4()),
    #         #     "name": config.node_manager.registry[g.nodes[node_name]["name"]],
    #         #     "position": {"x": 100, "y": 100},
    #         #     "fn": config.node_manager.registry[g.nodes[node_name]["fname"]],
    #         #     "width": 200,
    #         #     "height": 80,
    #         # }
    #         graph.nodes.append(node_data)
    #         # graph.links
    #         return
    #     except Exception as e:
    #         logger.exception(e)
    #         return {"error": str(e)}


def register_flow_routes(rtr: APIRouter):
    @rtr.get("/flows")
    def list_flows():
        return config.get_path_tree("flows")

    @rtr.get("/flows/{item}")
    def fetch_flow_item():
        return json.load(os.path.join(config.get_path("flows"), item))


def register_routes(rtr: APIRouter):
    register_node_routes(rtr)
    register_flow_routes(rtr)
