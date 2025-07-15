# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

# import json
# import uuid
# from fastapi import WebSocket
# from starlette.websockets import WebSocketDisconnect
# from typing import Dict
# from fastapi.responses import JSONResponse


from server.routes import register_update_signal


node_data = {
    "id": str(uuid.uuid4()),
    "name": "New Node",
    "position": {"x": 100, "y": 100},
    "fn": "some_node_function",
    "width": 200,
    "height": 80,
}

from fastapi import APIRouter,
async def add_node(websocket: WebSocket, action: str, data: dict):
    await send_command(websocket, "insert_node", node_data)


# @rtr.post("/command/insert_node")
# async def insert_node_api(node: Dict):
#     """
#     API endpoint to trigger insertion of a node.
#     This can be used if WebSockets are not preferred or for HTTP-based commands.
#     """
#     node_data = {
#         "id": str(uuid.uuid4()),
#         "name": "Display Text",
#         "position": {"x": 100, "y": 100},
#         "fn": "display_text",
#         "width": 200,
#         "height": 80,
#     }
#     # Simulate sending the command through WebSocket
#     # In a real implementation, you would broadcast to connected clients
#     return JSONResponse(content=node_data)


# async def send_command(websocket: WebSocket, action: str, data: dict):
#     message = json.dumps({"action": action, "data": data})
#     await websocket.send_text(message)


# # Example usage:
# node_data = {
#     "id": str(uuid.uuid4()),
#     "name": "Display Text",
#     "position": {"x": 100, "y": 100},
#     "fn": "display_text",
#     "width": 200,
#     "height": 80,
# }
# await send_command(websocket, "insert_node", node_data)

# from fastapi import APIRouter, WebSocket, WebSocketDisconnect
# from fastapi.responses import JSONResponse
# from typing import Dict
# import json
# import uuid


# def register_node_manipulation_routes(rtr: APIRouter):
#     @rtr.websocket("/ws/command")
#     async def command_websocket(websocket: WebSocket):
#         await websocket.accept()

#         # Helper function to send commands to the client
#         async def send_command(action: str, data: Dict):
#             message = {"action": action, "data": data}
#             await websocket.send_text(json.dumps(message))

#         try:
#             while True:
#                 # Wait for messages from the client (if needed)
#                 await websocket.receive_text()

#                 # Example: Send an "insert_node" command
#                 node_data = {
#                     # "id": str(uuid.uuid4()),  # Unique ID for the node
#                     "name": "New Node",  # Name of the node
#                     "position": {"x": 100, "y": 100},  # Initial position
#                     "fn": "some_node_function",  # Function associated with the node
#                     # "width": 200,  # Width of the node
#                     # "height": 80,  # Height of the node
#                 }
#                 await send_command("insert_node", node_data)

#         except WebSocketDisconnect:
#             pass

#     @rtr.post("/command/insert_node")
#     async def insert_node_api(node: Dict):
#         """
#         API endpoint to trigger insertion of a node.
#         This can be used if WebSockets are not preferred or for HTTP-based commands.
#         """
#         # Simulate sending the command through WebSocket
#         # In a real implementation, you would broadcast to connected clients
#         message = {"action": "insert_node", "data": node}
#         return JSONResponse(content=message)
