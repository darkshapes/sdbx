import pytest
import asyncio
import networkx as nx
from sdbx.executor import Executor, TaskContext
from sdbx.server.types import *
from sdbx.nodes.types import *
from unittest.mock import AsyncMock, MagicMock
import pytest_asyncio


@node(name="DumbAFNode", display=False)
def dummy_node_fn(x):
    return x + 1


dummy_node_fn.generator = False


@pytest.fixture
def mock_dummy_node_fn(monkeypatch):
    def mock_fn(x):
        return x * 2

    monkeypatch.setattr("test_executor.dummy_node_fn", mock_fn)
    return mock_fn


def test_dummy_node_fn(mock_dummy_node_fn):
    assert dummy_node_fn(2) == 4
    with pytest.raises(AssertionError):
        assert dummy_node_fn(2) == 3


# @pytest_asyncio.fixture(loop_scope="module")
# def dummy_node():
#     return Node(id="n1", fname="dummy_node_fn", widget_inputs={})


@pytest_asyncio.fixture(loop_scope="module")
async def mock_node_manager(mock_dummy_node_fn):
    context = TaskContext()
    with context.use():

        async def pulse_process_event():
            while not context.halt_event.is_set():
                context.process_event.set()
                await asyncio.sleep(0.05)

        asyncio.create_task(pulse_process_event())
        mock_dummy_node_fn.generator = False
        nm = MagicMock()
        nm.registry.return_value = {"dummy_node_fn": mock_dummy_node_fn}
        return nm


@pytest.mark.asyncio(loop_scope="module")
async def test_execute_node(mock_node_manager):
    executor = Executor(mock_node_manager)
    G = nx.MultiDiGraph()
    G.add_node("n1", fname="dummy_node_fn", widget_inputs={})
    context = TaskContext()
    with context.use():

        async def pulse_process_event():
            while not context.halt_event.is_set():
                context.process_event.set()
                await asyncio.sleep(0.05)

        asyncio.create_task(pulse_process_event())
        await executor.execute_node(node_id="n1", node={"id": "n1", "fname": "dummy_node_fn"}, inputs={"value": 5})
        mock_node_manager.registry["dummy_node_fn"].assert_called_once_with(value=5)
        with pytest.raises(AssertionError):
            mock_node_manager.registry["dummy_node_fn"].assert_called_once_with(value=4)


@pytest.mark.asyncio(loop_scope="module")
async def test_execute_node_unblocks(mock_node_manager, mock_dummy_node_fn):
    executor = Executor(mock_node_manager)
    context = TaskContext()
    with context.use():

        async def pulse_process_event():
            while not context.halt_event.is_set():
                context.process_event.set()
                await asyncio.sleep(0.05)

        asyncio.create_task(pulse_process_event())
        await executor.execute_node("n1", {"id": "n1", "fname": "dummy_node_fn"}, {"value": 7})
        print(context.results.get("n1"))
        mock_node_manager.registry["dummy_node_fn"].assert_called_once_with(value=7)
        with pytest.raises(AssertionError):
            mock_node_manager.registry["dummy_node_fn"].assert_called_once_with(value=8)


@pytest.mark.asyncio(loop_scope="module")
async def test_detect_cycles(mock_node_manager):
    executor = Executor(mock_node_manager)

    G = nx.MultiDiGraph()
    G.add_node("n1")
    G.add_node("n2")
    G.add_edge("n1", "n2")
    G.add_edge("n2", "n1")  # Creates a cycle
    context = TaskContext()
    with context.use():

        async def pulse_process_event():
            while not context.halt_event.is_set():
                context.process_event.set()
                await asyncio.sleep(0.05)

        asyncio.create_task(pulse_process_event())
        cycles = executor.detect_cycles(G)
        assert len(cycles) == 1
        assert set(cycles[0]) == {"n1", "n2"}


@pytest.mark.asyncio(loop_scope="module")
async def test_process_node(mock_node_manager):
    executor = Executor(mock_node_manager)
    context = TaskContext()

    G = nx.MultiDiGraph()
    G.add_node("n1", fname="dummy_node_fn", widget_inputs={"input": {"value": 1}})

    with context.use():
        # context.results["n1"] = (1,)
        context.process_event.set()

        async def pulse_process_event():
            while not context.halt_event.is_set():
                context.process_event.set()
                await asyncio.sleep(0.05)

        asyncio.create_task(pulse_process_event())
        await executor.process_node(G, node="n1")
        mock_node_manager.registry["dummy_node_fn"].assert_called_once_with(input=1)
        with pytest.raises(AssertionError):
            mock_node_manager.registry["dummy_node_fn"].assert_called_once_with(input=2)


# @pytest.mark.asyncio(loop_scope="module")
# async def test_execute_graph(mock_node_manager):
#     executor = Executor(mock_node_manager)
#     context = TaskContext()

#     G = nx.MultiDiGraph()
#     G.add_node("n1", fname="dummy_node_fn", widget_inputs={})

#     with context.use():

#         async def pulse_process_event():
#             while not context.halt_event.is_set():
#                 context.process_event.set()
#                 await asyncio.sleep(0.05)

#         asyncio.create_task(pulse_process_event())
#         results = await executor.execute_graph(G)

#         assert results.get("n1", 0) is not False
#         print(results["n1"])


# @pytest.mark.asyncio
# async def test_process_node_chain(mock_node_manager):
#     executor = Executor(mock_node_manager)
#     context = TaskContext()

#     G = nx.MultiDiGraph()
#     G.add_node("n1", id="n1", fname="dummy_node_fn", widget_inputs={"x": {"value": 10}})
#     G.add_node("n2", id="n2", fname="dummy_node_fn", widget_inputs={})
#     G.add_edge("n1", "n2", source_handle=0, target_handle="x")

#     with context.use():

#         async def pulse_process_event():
#             while not context.completion_event.is_set():
#                 context.process_event.set()
#                 await asyncio.sleep(0.05)

#         task_id = "t1"
#         asyncio.create_task(pulse_process_event())
#         executor.execute(G, task_id, context)

#         try:
#             await asyncio.wait_for(context.completion_event.wait(), timeout=3)
#         except asyncio.TimeoutError:
#             assert False, "Graph execution did not complete in time"
#         assert context.results["n1"] == (20,)
#         assert context.results["n2"] == (40,)
