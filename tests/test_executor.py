# from . import nodes # Using test nodes to test executor
import networkx as nx
import asyncio
from sdbx.executor import Executor, TaskContext
from sdbx.nodes.base.test import displays_string
from sdbx.nodes.base.nodes import outputs_string
from sdbx.nodes.manager import NodeManager
from sdbx.config import get_config_location
import os


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


async def test_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("Outputs String", fname="outputs_string", widget_inputs={"string": {"value": "printy printy test 🤡"}})  # <- matches
    graph.add_node("Displays String", fname="displays_string", inputs={"ouputs_string"})
    graph.add_edge("Outputs String", "Displays String")
    print(graph.graph)
    ext_loc = {"nodes": ""}
    node_loc = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")
    le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_loc)
    exec_inst = Executor(le_test_manager)
    task_id = "test_task"
    exec_inst.execute(graph, task_id)
    ext_loc = os.path.join(get_config_location(), "nodes")


#     graph = nx.MultiDiGraph()
#     graph.add_node("func_1", fname=outputs_string)
#     graph.add_node("func_2", fname=displays_string)
#     graph.add_edge("func_1", "func_2")
#     print(graph.graph)
#     ext_loc = {"nodes": ""}
#     # ext_loc = os.path.join(get_config_location(), "nodes")
#     node_loc = os.path.join(os.path.dirname(os.getcwd()), "sdbx", "nodes", "base")
#     # context = TaskContext()

#     le_test_manager = NodeManager(extensions=ext_loc, nodes_path=node_loc)
#     exec_inst = Executor(le_test_manager)
#     task_id = "test_task"
#     exec_inst.execute(graph, task_id)

#     # context.queue(await exec_inst.execute_graph(graph))


asyncio.run(test_graph())


# {
#   "nodes": {
#     "input_node": {
#       "fname": "input_string",
#       "widget_inputs": {
#         "input_value": {
#           "value": "Hello, World!"
#         }
#       }
#     },
#     "print_node": {
#       "fname": "print_string"
#     }
#   },
#   "edges": [
#     {
#       "source": "input_node",
#       "target": "print_node",
#       "source_handle": "output",
#       "target_handle": "input"
#     }
#   ]
# }
