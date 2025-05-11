import os
import numbers
from sdbx.indexer import IndexManager
from sdbx import logger
from sdbx.config import config, cache
from sdbx.nodes.helpers import soft_random
from collections import defaultdict


# def collect_tuned_parameters(self, node_manager, graph: MultiDiGraph, node_id: str):
#     predecessors = graph.predecessors(node_id)

#     node = graph.nodes[node_id]

#     tuned_parameters = {}
#     for p in predecessors:
#         pnd = graph.nodes[p]  # predecessor node data
#         pfn = node_manager.registry[pnd['fname']]  # predecessor function

#         p_tuned_parameters = pfn.tuner.get_tuned_parameters(pnd['widget_inputs'])[node['fname']]

#         tuned_parameters |= p_tuned_parameters

#     return tuned
# @cache
#     def get_tuned_parameters(self, widget_inputs, model_types, metadata):
#
#     def tuned_parameters(self, node_manager, graph: MultiDiGraph, node_id: str):
#         predecessors = graph.predecessors(node_id)
#         node = graph.nodes[node_id]
#         tuned_parameters = {}
#         for p in predecessors:
#             pnd = graph.nodes[p]  # predecessor node data
#             pfn = node_manager.registry[pnd['fname']]  # predecessor function
