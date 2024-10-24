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

from sdbx import config, logger
from sdbx.server.types import Edge, Node

current_context = contextvars.ContextVar('current_context')

class TaskContext:
    def __init__(self):
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
    def use(self):
        token = current_context.set(self)
        try:
            yield self
        finally:
            current_context.reset(token)

    @staticmethod
    def get_current():
        return current_context.get()

class Executor:
    def __init__(self, node_manager):
        self.node_manager = node_manager
        self.tasks = {}
    
    async def execute_node(self, node_id: str, node: Node, inputs: Dict[str, Any]):
        context = TaskContext.get_current()

        fname = node['fname']
        widget_inputs = node['widget_inputs']

        nf = self.node_manager.registry[fname] # Node function
        lf = partial(nf, **inputs, **widget_inputs) # Loaded function

        async def send_result(result):
            context.results[node_id] = result if isinstance(result, tuple) else (result,) # Ensure the output is iterable if isn't already
            context.result_event.set()

            await context.process_event.wait()
            context.process_event.clear()

            context.result_event.clear()

        g = lf()

        if nf.generator:
            for result in g:
                await send_result(result)
        else:
            await send_result(g)
    
    async def process_node(self, graph, node):
        context = TaskContext.get_current()

        # Get data of all input edges
        in_edge_data = [Edge(
            source=e[0],
            **v,
        ) for e in graph.in_edges(node) for v in graph.get_edge_data(*e).values()]

        # Gather inputs from predecessors
        # P.S. Do you see why we had to make the output iterable? :)
        inputs = { v.target_handle: context.results[v.source][v.source_handle] for v in in_edge_data }
        
        # Execute the node with the collected inputs
        await self.execute_node(node, graph.nodes[node], inputs)
        
        # Enqueue successors if they have received all their inputs
        for successor in graph.successors(node):
            if len(context.results[node]) == len(list(graph.predecessors(successor))):
                await context.queue.put(successor)
    
    def detect_cycles(self, graph):
        # Detect strongly connected components (SCCs)
        sccs = list(nx.strongly_connected_components(graph))
        
        # Filter out SCCs that are trivial (single node with no self-loop)
        cycles = [scc for scc in sccs if len(scc) > 1 or graph.has_edge(list(scc)[0], list(scc)[0])]
        
        return cycles
    
    async def handle_cycle(self, graph, cycle):
        context = TaskContext.get_current()

        # Process the cycle as a strongly connected component (SCC)
        subgraph = graph.subgraph(cycle)
        
        # Initialize node outputs
        for node in cycle:
            if node not in context.results:
                await self.execute_node(graph.nodes[node], [])
        
        # Iterate to propagate the feedback until halted
        while not context.halt_event.is_set():
            converged = True
            for node in cycle:
                inputs = [context.results[pred][-1] for pred in subgraph.predecessors(node)]
                output = await self.execute_node(graph.nodes[node], inputs)
                
                if not context.results[node] or output != context.results[node][-1]:
                    converged = False
                
                context.results[node] = output
                context.results.set()
            
            # If convergence is achieved or no change in results, break the loop
            if converged:
                break

    async def execute_graph(self, graph: MultiDiGraph):
        context = TaskContext.get_current()

        try:
            # Detect cycles in the graph
            cycles = self.detect_cycles(graph)
            
            # Setup cycle coroutines
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

            context.completion_event.set() # Completed execution successfully
            
            return context.results
        except Exception as e:
            # Capture the error and signal it via error_event
            logger.exception(e)
            context.task_error = e
            context.error_event.set()
    
    def execute(self, graph: MultiDiGraph, task_id: str):
        context = TaskContext()
        self.tasks[task_id] = context
        with context.use():
            context.running_task = create_task(self.execute_graph(graph))
    
    def halt(self, task_id: str):
        context = self.tasks.get(task_id)
        if context:
            context.halt_event.set()
            if context.running_task:
                context.running_task.cancel()