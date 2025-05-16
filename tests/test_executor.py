import pytest
import networkx as nx
from unittest.mock import MagicMock, patch

import pytest_asyncio
from sdbx.executor import Executor, TaskContext
from sdbx.nodes.types import *
from sdbx.server.types import Node
from nnll_01 import debug_monitor, dbug


@node(name="DumbAFNode", display=False)
def dummy_node_fn(x):
    return x + 1


dummy_node_fn.generator = False


@pytest.fixture
def mock_dummy_node_fn(monkeypatch):
    def mock_fn(x):
        return x * 2

    monkeypatch.setattr("tests.test_executor.dummy_node_fn", mock_fn)
    return mock_fn


def test_dummy_node_fn(mock_dummy_node_fn):
    assert dummy_node_fn(2) == 4
    with pytest.raises(AssertionError):
        assert dummy_node_fn(2) == 3


@pytest.fixture
def mock_dummy_node_fn_v2(monkeypatch):
    mock_dummy = MagicMock()
    monkeypatch.setattr("tests.test_executor.dummy_node_fn", mock_dummy)
    return mock_dummy


@pytest_asyncio.fixture(loop_scope="module")
def mock_node_manager(mock_dummy_node_fn_v2):
    node_manager = MagicMock()
    node_manager.registry.return_value = {"dummy_node_fn": mock_dummy_node_fn_v2}
    return node_manager


def test_mock_node_manager(mock_node_manager):
    node_manager = mock_node_manager
    node_manager.registry["dummy_node_fn"]("1")
    mock_node_manager.registry["dummy_node_fn"].assert_called_once_with("1")


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


@pytest.mark.asyncio(loop_scope="module")
async def test_execute_node(mock_node_manager):
    executor = Executor(mock_node_manager)
    context = TaskContext()
    with context.use():
        await executor.execute_node(node_id="n1", node={"id": "n1", "fname": "dummy_node_fn"}, inputs={})
        mock_node_manager.registry["dummy_node_fn"].assert_called_once_with()


@pytest.mark.asyncio(loop_scope="module")
async def test_process_node(mock_node_manager):
    executor = Executor(mock_node_manager)
    context = TaskContext()
    with context.use():
        G = nx.MultiDiGraph()
        G.add_node("n1")
        G.add_node("n2")
        G.add_edge("n1", "n2")
        node = Node(id="n1", fname="dummy_node_fn", widget_inputs={})
        print(G.nodes())
        await executor.process_node(G, node=G.nodes())
        assert "n1" in context.results


# @pytest.mark.asyncio(loop_scope="module")
# async def test_execute_graph(mock_node_manager):
#     executor = Executor(mock_node_manager)
#     context = TaskContext()
#     with context.use():
#         G = nx.MultiDiGraph()
#         G.add_node("n1")
#         G.add_node("n2")
#         G.add_edge("n1", "n2")
#         results = await executor.execute_graph(G)
#         assert "n1" in results
#         assert "n2" in results


# @pytest.mark.asyncio(loop_scope="module")
# async def test_execute(mock_node_manager):
#     executor = Executor(mock_node_manager)
#     G = nx.MultiDiGraph()
#     G.add_node("n1")
#     task_id = "test_task"
#     executor.execute(G, task_id)
#     assert task_id in executor.tasks
#     assert isinstance(executor.tasks[task_id], TaskContext)


# @pytest.mark.asyncio(loop_scope="module")
# async def test_halt(mock_node_manager):
#     executor = Executor(mock_node_manager)
#     G = nx.MultiDiGraph()
#     G.add_node("n1")
#     task_id = "test_task"
#     executor.execute(G, task_id)
#     executor.halt(task_id)
#     assert executor.tasks[task_id].halt_event.is_set()
