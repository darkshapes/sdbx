from typing import Any, Dict
from collections import defaultdict
from asyncio import Queue, Event, gather

import networkx as nx
from networkx import MultiDiGraph

from sdbx import config
from sdbx.server.types import Edge, Node

class Executor:
    def __init__(self, node_manager):
        self.node_manager = node_manager
        self.queue = Queue()
        self.results = defaultdict(list) # Store the results of each node execution
        self.halt_event = Event() # Event to signal halt
        self.result_event = Event() # Event to signal result updates
        self.running_task = None  # Task handle for the running executor
    
    async def execute_node(self, node: Node, inputs: Dict[str, Any]):
        return self.node_manager.registry[node['fname']](**inputs, **node['widget_inputs'])
    
    async def process_node(self, graph, node):
        # Get data of all input edges
        in_edge_data = [Edge(
            source=e[0],
            **v,
        ) for e in graph.in_edges(node) for v in graph.get_edge_data(*e).values()]

        # Gather inputs from predecessors
        inputs = { v.target_handle: self.results[v.source][v.source_handle] for v in in_edge_data }
        
        # Execute the node with the collected inputs
        output = await self.execute_node(graph.nodes[node], inputs)
        
        # Store the result
        self.results[node].append(output)

        # Notify that results have been updated
        self.result_event.set()
        
        # Enqueue successors if they have received all their inputs
        for successor in graph.successors(node):
            if len(self.results[node]) == len(list(graph.predecessors(successor))):
                await self.queue.put(successor)
    
    def detect_cycles(self, graph):
        # Detect strongly connected components (SCCs)
        sccs = list(nx.strongly_connected_components(graph))
        
        # Filter out SCCs that are trivial (single node with no self-loop)
        cycles = [scc for scc in sccs if len(scc) > 1 or graph.has_edge(list(scc)[0], list(scc)[0])]
        
        return cycles
    
    async def handle_cycle(self, graph, cycle):
        # Process the cycle as a strongly connected component (SCC)
        subgraph = graph.subgraph(cycle)
        
        # Initialize node outputs
        for node in cycle:
            if node not in self.results:
                await self.execute_node(graph.nodes[node], [])
        
        # Iterate to propagate the feedback until halted
        while not self.halt_event.is_set():
            converged = True
            for node in cycle:
                inputs = [self.results[pred][-1] for pred in subgraph.predecessors(node)]
                output = await self.execute_node(graph.nodes[node], inputs)
                
                if not self.results[node] or output != self.results[node][-1]:
                    converged = False
                
                self.results[node].append(output)
            
            # If convergence is achieved or no change in results, break the loop
            if converged:
                break

    async def execute(self, graph: MultiDiGraph):
        # Detect cycles in the graph
        cycles = self.detect_cycles(graph)
        
        # Start handling cycles
        tasks = [self.handle_cycle(graph, cycle) for cycle in cycles]
        
        # Initialize the queue with nodes that have no predecessors (input terminal nodes)
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                await self.queue.put(node)
        
        # Process nodes in topological order (acyclic parts)
        while not self.queue.empty() and not self.halt_event.is_set():
            node = await self.queue.get()
            await self.process_node(graph, node)
        
        # Wait for cycle handling tasks to complete
        await gather(*tasks)
        
        return self.results
    
    def halt(self):
        # Signal the executor to halt
        self.halt_event.set()
        
        # Optionally, cancel the running task if needed
        if self.running_task:
            self.running_task.cancel()