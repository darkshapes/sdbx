# from . import nodes # Using test nodes to test executor
import networkx as nx
import asyncio
from sdbx.executor import Executor, TaskContext
from sdbx.nodes.base.test import displays_string
from sdbx.nodes.base.nodes import outputs_string
from sdbx.nodes.manager import NodeManager
from sdbx.nodes.types import *

from sdbx.config import get_config_location
import os


async def executor_test():
    # Create a graph with the two nodes
    graph = nx.MultiDiGraph()
    graph.add_node("Input Node", fname="input_node", widget_inputs={"string": {"value": "Hello, World!"}})
    graph.add_node("Display Node", fname="display_node")
    graph.add_edge("Input Node", "Display Node", source_handle="0", target_handle="0")
    node_path = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")

    #     graph = nx.MultiDiGraph(graph_data)
    #     exec_inst.execute(graph, task_id)  # context = TaskContext()

    # Initialize the node manager and executor
    node_manager = NodeManager(extensions={"nodes": {}}, nodes_path=node_path)
    exe = Executor(node_manager)

    # Execute the graph with a test string
    exe.execute(graph, "test_task")
    await exe.tasks["test_task"].queue.put("Input Node")
    exe.tasks["test_task"].process_event.set()

    # Run the test with a sample string
    exe.tasks["test_task"].results["Input Node"] = ("Hello, world!",)
    exe.tasks["test_task"].result_event.set()
    await exe.tasks["test_task"].process_event.wait()
    exe.tasks["test_task"].process_event.clear()


# async def test_graph():
#     graph = nx.MultiDiGraph()
#     graph.add_node("Outputs String", fname="outputs_string", widget_inputs={"string": {"value": "printy printy test 🤡"}})  # <- matches
#     graph.add_node("Displays String", fname="displays_string", inputs={"ouputs_string"})
#     graph.add_edge("Outputs String", "Displays String")
#     print(graph.graph)
#     ext_loc = {"nodes": ""}
#     node_path = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")
#     le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_path)
#     exec_inst = Executor(le_test_manager)
#     task_id = "test_task"
#     exec_inst.execute(graph, task_id)
#     # ext_loc = os.path.join(get_config_location(), "nodes")


# async def test_graph_prestructured():
#     graph_data = {
#         "nodes": {"input_node": {"fname": "outputs_string", "widget_inputs": {"input_value": {"value": "Hello, World!"}}}, "print_node": {"fname": "displays_string"}},
#         "edges": [{"source": "input_node", "target": "print_node", "source_handle": "output", "target_handle": "input"}],
#     }
#     graph = nx.MultiDiGraph(graph_data)
#     print(graph.graph)
#     ext_loc = {"nodes": ""}
#     node_path = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")
#     le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_path)
#     exec_inst = Executor(le_test_manager)
#     task_id = "test_task"

# ext_loc = os.path.join(get_config_location(), "nodes")


# graph = nx.MultiDiGraph()
# graph.add_node("Outputs String", fname=outputs_string)
# graph.add_node("Displays String", fname=displays_string)
# graph.add_edge("Outputs String", "Displays String")
# print(graph.graph)
# ext_loc = {"nodes": ""}
# node_loc = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")
# le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_loc)

# context = TaskContext()

# exec_inst = Executor(le_test_manager)
# task_id = "test_task"
# exec_inst.execute(graph, task_id)
# ext_loc = os.path.join(get_config_location(), "nodes")

# # context.queue(await exec_inst.execute_graph(graph))


#     le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_loc)
#     exec_inst = Executor(le_test_manager)
#     task_id = "test_task"
#     exec_inst.execute(graph, task_id)

#     # context.queue(await exec_inst.execute_graph(graph))


asyncio.run(executor_test())
