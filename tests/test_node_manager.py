import pytest
import networkx as nx
from unittest.mock import MagicMock, patch

import pytest_asyncio
from sdbx.executor import Executor, TaskContext
from sdbx.nodes.types import *
from sdbx.server.types import Node
from nnll.monitor.file import debug_monitor, dbug


@node(name="DumbAFNode", display=False)
def dummy_node_fn(x):
    return x + 1


dummy_node_fn.generator = False


@pytest.fixture
def mock_dummy_node_fn(monkeypatch):
    def mock_fn(x):
        return x * 2

    monkeypatch.setattr("test_node_manager.dummy_node_fn", mock_fn)
    return mock_fn


def test_dummy_node_fn(mock_dummy_node_fn):
    assert dummy_node_fn(2) == 4
    with pytest.raises(AssertionError):
        assert dummy_node_fn(2) == 3


@pytest.fixture
def mock_dummy_node_fn_v2(monkeypatch):
    mock_dummy = MagicMock()
    monkeypatch.setattr("test_node_manager.dummy_node_fn", mock_dummy)
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
