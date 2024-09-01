from __future__ import annotations

import asyncio
import gc
import uuid
from asyncio import get_event_loop
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..api.components.schema.prompt import PromptDict
from ..component_model.executor_types import ExecutorToClientProgress
from ..component_model.make_mutable import make_mutable
from ..distributed.server_stub import ServerStub

import asyncio
import uuid
from asyncio import AbstractEventLoop
from collections import defaultdict
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse, urljoin

import aiohttp
from aiohttp import WSMessage, ClientResponse
from typing_extensions import Dict

from sdbx.config import config, Config

from ..api.api_client import JSONEncoder
from ..api.components.schema.prompt import PromptDict
from ..api.components.schema.prompt_request import PromptRequest
from ..api.paths.history.get.responses.response_200.content.application_json.schema import Schema as GetHistoryDict
from ..api.schemas import immutabledict

_server_stub_instance = ServerStub()

import dataclasses
from typing import List

from typing_extensions import TypedDict, Literal, NotRequired, Dict


class FileOutput(TypedDict, total=False):
    filename: str
    subfolder: str
    type: Literal["output", "input", "temp"]
    abs_path: str
    name: NotRequired[str]


class Output(TypedDict, total=False):
    latents: NotRequired[List[FileOutput]]
    images: NotRequired[List[FileOutput]]


@dataclasses.dataclass
class V1QueuePromptResponse:
    urls: List[str]
    outputs: Dict[str, Output]
    

class AsyncRemoteShadowboxClient:
    """
    An asynchronous client for remote servers
    """
    __json_encoder = JSONEncoder()

    def __init__(self, server_address: str = "http://localhost:8188", client_id: str = str(uuid.uuid4()),
                 websocket_address: Optional[str] = None, loop: Optional[AbstractEventLoop] = None):
        self.client_id = client_id
        self.server_address = server_address
        server_address_url = urlparse(server_address)
        self.websocket_address = websocket_address if websocket_address is not None else urljoin(
            f"ws://{server_address_url.hostname}:{server_address_url.port}", f"/ws?clientId={client_id}")
        self.loop = loop or asyncio.get_event_loop()

    async def len_queue(self) -> int:
        async with aiohttp.ClientSession() as session:
            async with session.get(urljoin(self.server_address, "/prompt"), headers={'Accept': 'application.json'}) as response:
                if response.status == 200:
                    exec_info_dict = await response.json()
                    return exec_info_dict["exec_info"]["queue_remaining"]
                else:
                    raise RuntimeError(f"unexpected response: {response.status}: {await response.text()}")

    async def queue_prompt_api(self, prompt: PromptDict) -> V1QueuePromptResponse:
        """
        Calls the API to queue a prompt.
        :param prompt:
        :return: the API response from the server containing URLs and the outputs for the UI (nodes with OUTPUT_NODE == true)
        """
        prompt_json = AsyncRemoteShadowboxClient.__json_encoder.encode(prompt)
        async with aiohttp.ClientSession() as session:
            response: ClientResponse
            async with session.post(urljoin(self.server_address, "/api/v1/prompts"), data=prompt_json,
                                    headers={'Content-Type': 'application/json', 'Accept': 'application/json'}) as response:

                if response.status == 200:
                    return V1QueuePromptResponse(**(await response.json()))
                else:
                    raise RuntimeError(f"could not prompt: {response.status}: {await response.text()}")

    async def queue_prompt_uris(self, prompt: PromptDict) -> List[str]:
        """
        Calls the API to queue a prompt.
        :param prompt:
        :return: a list of URLs corresponding to the SaveImage nodes in the prompt.
        """
        return (await self.queue_prompt_api(prompt)).urls

    async def queue_prompt(self, prompt: PromptDict) -> bytes:
        """
        Calls the API to queue a prompt. Returns the bytes of the first PNG returned by a SaveImage node.
        :param prompt:
        :return:
        """
        prompt_json = AsyncRemoteShadowboxClient.__json_encoder.encode(prompt)
        async with aiohttp.ClientSession() as session:
            response: ClientResponse
            async with session.post(urljoin(self.server_address, "/api/v1/prompts"), data=prompt_json,
                                    headers={'Content-Type': 'application/json', 'Accept': 'image/png'}) as response:

                if 200 <= response.status < 400:
                    return await response.read()
                else:
                    raise RuntimeError(f"could not prompt: {response.status}: {await response.text()}")

    async def queue_prompt_ui(self, prompt: PromptDict) -> Dict[str, List[Path]]:
        """
        Uses the sdbx UI API calls to retrieve a list of paths of output files
        :param prompt:
        :return:
        """
        prompt_request = PromptRequest.validate({"prompt": prompt, "client_id": self.client_id})
        prompt_request_json = AsyncRemoteShadowboxClient.__json_encoder.encode(prompt_request)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.websocket_address) as ws:
                async with session.post(urljoin(self.server_address, "/prompt"), data=prompt_request_json,
                                        headers={'Content-Type': 'application/json'}) as response:
                    if response.status == 200:
                        prompt_id = (await response.json())["prompt_id"]
                    else:
                        raise RuntimeError("could not prompt")
                msg: WSMessage
                async for msg in ws:
                    # Handle incoming messages
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        msg_json = msg.json()
                        if msg_json["type"] == "executing":
                            data = msg_json["data"]
                            if data['node'] is None and data['prompt_id'] == prompt_id:
                                break
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        break
            async with session.get(urljoin(self.server_address, "/history")) as response:
                if response.status == 200:
                    history_json = immutabledict(GetHistoryDict.validate(await response.json()))
                else:
                    raise RuntimeError("Couldn't get history")

            # images have filename, subfolder, type keys
            # todo: use the OpenAPI spec for this when I get around to updating it
            outputs_by_node_id = history_json[prompt_id].outputs
            res: Dict[str, List[Path]] = {}
            for node_id, output in outputs_by_node_id.items():
                if 'images' in output:
                    images = []
                    image_dicts: List[dict] = output['images']
                    for image_file_output_dict in image_dicts:
                        image_file_output_dict = defaultdict(None, image_file_output_dict)
                        filename = image_file_output_dict['filename']
                        subfolder = image_file_output_dict['subfolder']
                        type = image_file_output_dict['type']
                        images.append(Path(file_output_path(filename, subfolder=subfolder, type=type)))
                    res[node_id] = images
            return res


class EmbeddedShadowboxClient:
    """
    Embedded client for sdbx executing prompts as a library.

    This client manages a single-threaded executor to run long-running or blocking tasks
    asynchronously without blocking the asyncio event loop. It initializes a PromptExecutor
    in a dedicated thread for executing prompts and handling server-stub communications.

    Example usage:

    Asynchronous (non-blocking) usage with async-await:
    ```
    # Write a workflow, or enable Dev Mode in the UI settings, then Save (API Format) to get the workflow in your
    # workspace.
    prompt_dict = {
      "1": {"class_type": "KSamplerAdvanced", ...}
      ...
    }

    # Validate your workflow (the prompt)
    from sdbx.api.components.schema.prompt import Prompt
    prompt = Prompt.validate(prompt_dict)
    # Then use the client to run your workflow. This will start, then stop, a local sdbx workflow executor.
    # It does not connect to a remote server.
    async def main():
        async with EmbeddedShadowboxClient() as client:
            outputs = await client.queue_prompt(prompt)
            print(outputs)
        print("Now that we've exited the with statement, all your VRAM has been cleared from sdbx")

    if __name__ == "__main__"
        asyncio.run(main())
    ```

    In order to use this in blocking methods, learn more about asyncio online.
    """

    def __init__(self, configuration: Optional[Config] = None, progress_handler: Optional[ExecutorToClientProgress] = None, max_workers: int = 1):
        self._progress_handler = progress_handler or ServerStub()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._configuration = configuration
        # we don't want to import the executor yet
        self._prompt_executor: Optional["sdbx.cmd.execution.PromptExecutor"] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    async def __aenter__(self):
        await self._initialize_prompt_executor()
        self._is_running = True
        return self

    async def __aexit__(self, *args):
        # Perform cleanup here
        def cleanup():
            from .. import model_management
            model_management.unload_all_models()
            gc.collect()
            try:
                model_management.soft_empty_cache()
            except:
                pass

        # wait until the queue is done
        while self._executor._work_queue.qsize() > 0:
            await asyncio.sleep(0.1)

        await get_event_loop().run_in_executor(self._executor, cleanup)

        self._executor.shutdown(wait=True)
        self._is_running = False

    async def _initialize_prompt_executor(self):
        # This method must be async since it's used in __aenter__
        def create_executor_in_thread():
            from .. import options
            if self._configuration is None:
                options.enable_args_parsing()
            else:
                from ..cmd import args
                args.clear()
                args.update(self._configuration)

            from ..cmd import PromptExecutor

            self._prompt_executor = PromptExecutor(self._progress_handler)
            self._prompt_executor.raise_exceptions = True

        await get_event_loop().run_in_executor(self._executor, create_executor_in_thread)

    async def queue_prompt(self,
                           prompt: PromptDict | dict,
                           prompt_id: Optional[str] = None,
                           client_id: Optional[str] = None) -> dict:
        prompt_id = prompt_id or str(uuid.uuid4())
        client_id = client_id or self._progress_handler.client_id or None

        def execute_prompt() -> dict:
            from ..cmd import PromptExecutor, validate_prompt
            prompt_mut = make_mutable(prompt)
            validation_tuple = validate_prompt(prompt_mut)
            if not validation_tuple[0]:
                validation_error_dict = validation_tuple[1] or {"message": "Unknown", "details": ""}
                raise ValueError("\n".join([validation_error_dict["message"], validation_error_dict["details"]]))

            prompt_executor: PromptExecutor = self._prompt_executor

            if client_id is None:
                prompt_executor.server = _server_stub_instance
            else:
                prompt_executor.server = self._progress_handler

            prompt_executor.execute(prompt_mut, prompt_id, {"client_id": client_id},
                                    execute_outputs=validation_tuple[2])
            return prompt_executor.outputs_ui

        return await get_event_loop().run_in_executor(self._executor, execute_prompt)
