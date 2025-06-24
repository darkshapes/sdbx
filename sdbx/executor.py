import inspect
import logging
import json
import contextvars

from itertools import tee
from typing import Any, Dict
from functools import partial
from collections import defaultdict
from contextlib import contextmanager
from asyncio import Queue, Event, create_task, gather

import networkx as nx
from networkx import MultiDiGraph
from nnll.monitor.file import debug_monitor
from sdbx import config, logger
from sdbx.nodes.manager import NodeManager
from sdbx.server.types import Edge, Node


current_context = contextvars.ContextVar("current_context")


class TaskContext:
    """Hold context of tasks\n"""

    def __init__(self) -> None:
        """Create queue, results, task error, running task\n
        Event handlers for halt/result/error/process/completion"""
        self.queue = Queue()
        self.results = defaultdict()
        self.halt_event = Event()
        self.result_event = Event()
        self.error_event = Event()
        self.process_event = Event()
        self.completion_event = Event()
        self.task_error: Exception = None
        self.running_task = None

    @contextmanager
    def use(self) -> Any:
        """Set context and restore previous context after execution"""
        token = current_context.set(self)
        try:
            yield self
        finally:
            current_context.reset(token)

    @staticmethod
    def get_current() -> Queue | defaultdict | Event | None:
        """The currently active `TaskContext`"""
        return current_context.get()


class Executor:
    """Execute graph nodes, interface with `node_manager` and track active tasks"""

    def __init__(self, node_manager: NodeManager) -> None:
        self.node_manager = node_manager
        self.tasks = {}

    # @debug_monitor
    async def execute_node(self, node_id: str, node: Node, inputs: Dict[str, Any]) -> None:
        """Retrieve function name and processes additional inputs from\n
        :param node_id: Title of the node
        :param node: Dictionary containing node functions (expects "fname" key)
        :param inputs: Arguments collected from the node widgets to feed into the node function
        """
        context = TaskContext.get_current()

        fname = node["fname"]
        widget_inputs = {name: data.get("value") for name, data in node.get("widget_inputs", {}).items()}
        # why is the client sending all of this data? doesn't the executor just need the value?
        # are we doing validation on the server side?

        nf = self.node_manager.registry[fname]  # Node function
        lf = partial(nf, **inputs, **widget_inputs)  # Loaded function (with partially pre-filled arguments)

        async def send_result(result) -> None:
            """Store node’s output in context under node name, signal result is available\n
            Wait for a processing event to ensure synchronization\n
            :param result: The node function output
            """
            context.results[node_id] = result if isinstance(result, tuple) else (result,)  # Ensure the output is iterable if isn't already
            context.result_event.set()

            await context.process_event.wait()
            context.process_event.clear()

            context.result_event.clear()

        g = lf()  # the partially applied function pre-filled with values from `inputs` and `widget_inputs`.

        if nf.generator:
            for result in g:
                await send_result(result)
        else:
            await send_result(g)  # Or just send the final result

    async def process_node(self, graph: nx.MultiDiGraph, node: Node) -> None:
        """Store node execution `result` as tuple within task context `context.results` under `node_id`.\n
        :param graph: Refers to the graph in use
        :param node: Dictionary containing node functions
        """

        context = TaskContext.get_current()

        # Gather metadata from all input edges connecting to `node`
        in_edge_data = [
            Edge(
                source=e[0],
                **v,
            )
            for e in graph.in_edges(node)
            for v in graph.get_edge_data(*e).values()
        ]

        # Construct dictionary of function arguments, including predecessor nodes
        # P.S. Do you see why we had to make the output iterable? :)
        inputs = {v.target_handle: context.results[v.source][v.source_handle] for v in in_edge_data}

        # Concurrently execute the node with the collected inputs
        await self.execute_node(node, graph.nodes[node], inputs)

        # Verify all predecessors have results, then enqueue successors with predecessor inputs
        for successor in graph.successors(node):
            if len(context.results[node]) == len(list(graph.predecessors(successor))):
                await context.queue.put(successor)

    def detect_cycles(self, graph: nx.MultiDiGraph) -> list:
        """The subgraph of completely reachable cycling components\n
        :param graph: The working graph
        :return: A list of graph edges that should be cyclically executed
        """
        # Detect strongly connected components (SCCs), ie:
        sccs = list(nx.strongly_connected_components(graph))

        # Filter out SCCs that are trivial (single node with no self-loop)
        cycles = [scc for scc in sccs if len(scc) > 1 or graph.has_edge(list(scc)[0], list(scc)[0])]

        return cycles

    async def handle_cycle(self, graph: nx.MultiDiGraph, cycle: list):
        """Execution pattern to handle cycle repetition\n
        :param graph: Refers to the graph in use
        :param cycle: A predetermined list of nodes that loop which should be run
        """
        context = TaskContext.get_current()

        # Process the cycle as a strongly connected component (SCC)
        subgraph = graph.subgraph(cycle)

        # Initialize node outputs
        for node in cycle:
            if node not in context.results:  #  note: empty list input
                await self.execute_node(graph.nodes[node], [])  # pylint: disable=no-value-for-parameter

        # Iterate to propagate the feedback until halted
        while not context.halt_event.is_set():
            converged = True
            for node in cycle:
                inputs = [context.results[pred][-1] for pred in subgraph.predecessors(node)]  #  note: inputs here from previous nodes
                output = await self.execute_node(graph.nodes[node], inputs)  # pylint: disable=no-value-for-parameter

                if not context.results[node] or output != context.results[node][-1]:
                    converged = False

                context.results[node] = output
                context.results.set()

            # If convergence is achieved or no change in results, break the loop
            if converged:
                break

    async def execute_graph(self, graph: MultiDiGraph):
        """Initialize queue, gather nodes, then execute in sequence or by loop\n
        :param graph: Refers to the graph in use
        :return: Results of the function executions
        """
        context = TaskContext.get_current()

        try:
            # Detect cycles that run multiple times in the graph
            cycles = self.detect_cycles(graph)

            # Process cycle coroutines
            ct = [self.handle_cycle(graph, cycle) for cycle in cycles]

            # Initialize the queue with nodes that have no predecessors (input terminal nodes)
            for node in graph.nodes:
                if graph.in_degree(node) == 0:
                    await context.queue.put(node)

            # Process nodes in topological order (acyclic parts)
            while not context.queue.empty() and not context.halt_event.is_set():
                node = await context.queue.get()
                await self.process_node(graph, node)

            # Wait for cycle handling coroutines to complete
            await gather(*ct)

            context.completion_event.set()  # Completed execution successfully

            return context.results
        except Exception as e:
            # Capture the error and signal it via error_event
            logger.exception(e)
            context.task_error = e
            context.error_event.set()

    def execute(self, graph: MultiDiGraph, task_id: str, context: TaskContext = None) -> None:
        """Start the execution process of the specified graph\n
        :param graph: Refers to the graph in use
        :param task_id: The context indicator for the execution
        """
        context = context or TaskContext()
        self.tasks[task_id] = context
        with context.use():
            context.running_task = create_task(self.execute_graph(graph))

    def halt(self, task_id: str):
        """End the execution process of the specified task\n
        :param task_id: The context indicator being executed
        """
        context = self.tasks.get(task_id)
        if context:
            context.halt_event.set()
            if context.running_task:
                context.running_task.cancel()
