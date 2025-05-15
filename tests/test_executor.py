# test_executor.py
# test_executor.py

import pytest
import networkx as nx
from unittest.mock import MagicMock, patch

import pytest_asyncio
from sdbx.executor import Executor, TaskContext
from sdbx.server.types import *  # Adjust path if needed
from nnll_01 import debug_monitor, dbug

# Simulate a decorated function with the expected attributes


def dummy_node_fn(x):
    return x + 1


dummy_node_fn.generator = False


class DummyNodeInfo:
    def __init__(self, fn, **kwargs):
        self.name = fn.__name__

    def dict(self):
        return {"name": "dummy_node_fn"}


dummy_node_fn.info = DummyNodeInfo(dummy_node_fn)


@pytest_asyncio.fixture(loop_scope="module")
def mock_node_manager():
    node_manager = MagicMock()
    node_manager.registry = {"dummy_node_fn": dummy_node_fn}
    return node_manager


@pytest.mark.asyncio(loop_scope="module")
async def test_execute_simple_node(mock_node_manager):
    executor = Executor(mock_node_manager)

    # Create a simple graph with one node
    G = nx.MultiDiGraph()
    G.add_node("n1", fname="dummy_node_fn", widget_inputs={"x": {"value": 1}})
    dbug("executor_async", executor)
    dbug("graph_async", G)
    context = TaskContext()
    with context.use():
        dbug("context", context)
        await executor.execute_graph(G)
        assert context.results["n1"] == (2,)  # 1 + 1


@pytest.mark.asyncio(loop_scope="module")
async def test_execute_graph_with_dependencies(mock_node_manager):
    executor = Executor(mock_node_manager)

    # Create a graph with two dependent nodes
    G = nx.MultiDiGraph()
    G.add_node("n1", fname="dummy_node_fn", widget_inputs={"x": {"value": 2}})
    G.add_node("n2", fname="dummy_node_fn", widget_inputs={})
    G.add_edge("n1", "n2", source_handle=0, target_handle="x")
    dbug("graph_async", G)
    dbug("executor_async", executor)
    context = TaskContext()
    with context.use():
        await executor.execute_graph(G)
        dbug("context", context)
        assert context.results["n1"] == (3,)  # 2 + 1
        assert context.results["n2"] == (4,)  # 3 + 1


@debug_monitor
@pytest.mark.asyncio(loop_scope="module")
async def test_detect_cycles(mock_node_manager):
    executor = Executor(mock_node_manager)

    G = nx.MultiDiGraph()
    G.add_node("n1")
    G.add_node("n2")
    G.add_edge("n1", "n2")
    G.add_edge("n2", "n1")  # Creates a cycle

    cycles = executor.detect_cycles(G)
    assert len(cycles) == 1
    assert set(cycles[0]) == {"n1", "n2"}


# import pytest
# import asyncio
# import networkx as nx
# from unittest import mock

# from sdbx.executor import Executor, TaskContext
# from sdbx.config import ExtensionRegistry
# from sdbx.nodes.types import *
# from sdbx.nodes.manager import NodeManager


# # @node(name="InputNode", display=False, path="test")
# # def input_node(input_string: str):
# #     return input_string


# # @node(name="DisplayNode", display=True, path="test")
# # def display_node(message: str):
# #     print(f"Received: {message}")
# #     return message


# @pytest.fixture
# @node(name="InputNode", display=False, path="test")
# def mock_input_node():
#     with mock.patch(__name__ + ".input_node") as mock_func:
#         yield mock_func


# @pytest.fixture
# @node(name="DisplayNode", display=True, path="test")
# def mock_display_node():
#     with mock.patch(__name__ + ".display_node") as mock_func:
#         yield mock_func


# def test_input_node(mock_input_node):
#     test_string = "test_value"
#     mock_input_node.return_value = test_string

#     result = mock_input_node(test_string)

#     mock_input_node.assert_called_once_with(test_string)
#     assert result == test_string


# def test_display_node(mock_display_node):
#     test_message = "test_value"
#     mock_display_node.return_value = test_message

#     result = mock_display_node(test_message)

#     mock_display_node.assert_called_once_with(test_message)
#     assert result == test_message


# @pytest.fixture
# def mock_registry(mock_input_node, mock_display_node):
#     registry = {"input_node": mock_input_node, "display_node": mock_display_node}
#     return registry


# @pytest.mark.asyncio
# async def test_string_pass_through(mock_input_node, mock_display_node, mock_registry):
#     import os

#     G = nx.MultiDiGraph()

#     # Node definitions in graph
#     G.add_node("input", fname="input_node", widget_inputs={"string": {"value": "hello"}})
#     G.add_node("display", fname="display_node", widget_inputs={})

#     G.add_edge("input", "display", source_handle=0, target_handle="message")
#     node_path = os.path.join(os.path.dirname(os.getcwd()), "test")
#     node_manager = mock_registry

#     executor = Executor(node_manager=node_manager)

#     # context = TaskContext()
#     # with context.use():
#     executor.execute(G, task_id="test")
#     # print(context.results)
#     executor.halt("task_id")
#     mock_input_node.assert_called_once()
#     mock_display_node.assert_called_once()

#     return


# === Define Real Nodes (Used in Graph Execution) ===

# === Mocks for Unit Test ===

# === Test Unit Invocation (Not Graph Execution) ===

# === Integration Test of Graph Execution ===


# import pytest
# import asyncio
# import networkx as nx
# from sdbx.nodes.types import *
# from sdbx.executor import Executor, TaskContext
# from sdbx.nodes.manager import NodeManager
# from unittest import mock


# @pytest.fixture
# def mock_input_node():
#     with mock.patch("sdbx.nodes.base.test.input_node", new=mock.Mock()) as mock_func:
#         yield mock_func


# def test_input_node(mock_input_node):
#     from sdbx.nodes.base.test import input_node

#     test_string = "test_value"
#     mock_input_node.return_value = test_string
#     result = input_node(test_string)
#     mock_input_node.assert_called_once_with(test_string)
#     assert result == test_string


# @pytest.fixture
# def mock_display_node():
#     with mock.patch("sdbx.nodes.base.test.display_node", new=mock.Mock()) as mock_func:
#         yield mock_func


# def test_display_node(mock_display_node):
#     from sdbx.nodes.base.test import display_node

#     mock_display_node.return_value = mock_input_node
#     result = display_node(mock_input_node)
#     mock_display_node.assert_called_once_with(mock_input_node)
#     assert result == mock_input_node


# @node(name="InputNode", display=False, path="test")
# def input_node(input_string: str):
#     return input_string


# @node(name="DisplayNode", display=True, path="test")
# def display_node(message: str):
#     print(f"Received: {message}")
#     return message


# class MockNodeManager:
#     def __init__(self):
#         self.registry = {"input_node": input_node, "display_node": display_node}


# @pytest.mark.asyncio
# async def test_string_pass_through(mock_input_node, mock_display_node):
#     G = nx.MultiDiGraph()
#     G.add_node("input", fname="input_node", widget_inputs={"input_string": {"value": "hello"}})
#     G.add_node("display", fname="display_node", widget_inputs={})
#     G.add_edge("input_node", "display_node", source_handle=0, target_handle="message")

#     executor = Executor(node_manager=MockNodeManager())
#     results = executor.execute(G, "test_task")
#     mock_input_node.assert_called_once_with("hello")
#     mock_display_node.assert_called_once_with("hello")

#     assert results["display"] == ("hello",)


# # from . import nodes # Using test nodes to test executor
# import networkx as nx
# import asyncio
# from sdbx.executor import Executor, TaskContext
# from sdbx.nodes.manager import NodeManager
# from sdbx.nodes.types import *
# from sdbx.config import ExtensionData, ExtensionRegistry, get_config_location
# import os

# import unittest
# import pytest
# from unittest import mock
# import pytest


# @pytest.fixture
# def mock_file_content():
#     return f"{mock_input_node}\n{mock_display_node}"


# @pytest.fixture
# def mock_file(mock_file_content):
#     with mock.patch("builtins.open", mock.mock_open(read_data=mock_file_content)):
#         yield


# @pytest.mark.asyncio(loop_scope="module")
# async def test_executor(mock_input_node, mock_display_node):
#     # Create a graph with the two nodes
#     # graph = nx.MultiDiGraph()
#     # graph.add_node("Input Node", fname="input_node", widget_inputs={"string": {"value": "🤡"}})
#     # graph.add_node("Display Node", fname="display_node")
#     # graph.add_edge("Input Node", "Display Node", source_handle="0", target_handle="0")

#     node_manager = NodeManager(node_modules=ExtensionRegistry, nodes_path=node_path)
#     exe = Executor(node_manager)
#     print(node_manager.registry)

#     # exe.execute(graph, "test_task")

#     mock_input_node.assert_called_once_with("🤡")
#     mock_display_node.assert_called_once_with("🤡")

#     # await exe.tasks["test_task"].queue.put("Input Node")
#     # exe.tasks["test_task"].process_event.set()

#     # # Run the test with a sample string
#     # exe.tasks["test_task"].results["Input Node"] = ("Hello, world!",)
#     # exe.tasks["test_task"].result_event.set()
#     # await exe.tasks["test_task"].process_event.wait()
#     # exe.tasks["test_task"].process_event.clear()

#     #     graph = nx.MultiDiGraph(graph_data)
#     #     exec_inst.execute(graph, task_id)  # context = TaskContext()
#     # Execute the graph with a test string

#     # Initialize the node manager and executor


# # async def test_graph():
# #     graph = nx.MultiDiGraph()
# #     graph.add_node("Outputs String", fname="outputs_string", widget_inputs={"string": {"value": "printy printy test 🤡"}})  # <- matches
# #     graph.add_node("Displays String", fname="displays_string", inputs={"ouputs_string"})
# #     graph.add_edge("Outputs String", "Displays String")
# #     print(graph.graph)
# #     ext_loc = {"nodes": ""}
# #     node_path = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")
# #     le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_path)
# #     exec_inst = Executor(le_test_manager)
# #     task_id = "test_task"
# #     exec_inst.execute(graph, task_id)
# #     # ext_loc = os.path.join(get_config_location(), "nodes")


# # async def test_graph_prestructured():
# #     graph_data = {
# #         "nodes": {"input_node": {"fname": "outputs_string", "widget_inputs": {"input_value": {"value": "Hello, World!"}}}, "print_node": {"fname": "displays_string"}},
# #         "edges": [{"source": "input_node", "target": "print_node", "source_handle": "output", "target_handle": "input"}],
# #     }
# #     graph = nx.MultiDiGraph(graph_data)
# #     print(graph.graph)
# #     ext_loc = {"nodes": ""}
# #     node_path = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")
# #     le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_path)
# #     exec_inst = Executor(le_test_manager)
# #     task_id = "test_task"

# # ext_loc = os.path.join(get_config_location(), "nodes")


# # graph = nx.MultiDiGraph()
# # graph.add_node("Outputs String", fname=outputs_string)
# # graph.add_node("Displays String", fname=displays_string)
# # graph.add_edge("Outputs String", "Displays String")
# # print(graph.graph)
# # ext_loc = {"nodes": ""}
# # node_loc = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")
# # le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_loc)

# # context = TaskContext()

# # exec_inst = Executor(le_test_manager)
# # task_id = "test_task"
# # exec_inst.execute(graph, task_id)
# # ext_loc = os.path.join(get_config_location(), "nodes")

# # # context.queue(await exec_inst.execute_graph(graph))


# #     le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_loc)
# #     exec_inst = Executor(le_test_manager)
# #     task_id = "test_task"
# #     exec_inst.execute(graph, task_id)

# #     # context.queue(await exec_inst.execute_graph(graph))
