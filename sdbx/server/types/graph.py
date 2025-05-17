    """
    """
from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class Edge(BaseModel):
    """Represents a connection between two points in the graph\n
    #### `Edge` [{0 : {*source*:"",*source_handle*:"",*target_handle*:""}}]\n
    - `source`: `str` The origin node for the edge.
    - `source_handle`: `int` Output port on the node.
    - `target_handle`: `str`identifier for input port on the target node.
    """

    source: str
    source_handle: int
    target_handle: str


class Node(BaseModel):
    """A computational unit on the graph.\n
    #### `Node` [ name: {*id*:"",*fname*:"",*outputs*:[""],*inputs*:[""],*widget_inputs*:{"":*},}]\n
    - `id`: `str` Unique node identifier.
    - `fname`: `str` Function associated with the node.
    - `outputs`: `list[str]` Output ports of the node.
    - `inputs`: `list[str]` Input ports of the node.
    - `widget_inputs`dict[str:any]` Mapping of input name keys to previously calculated results.
    """
    # widget inputs dict[**str**: any].  is `str` the node name? the edge source? the target handle?

    id: str
    fname: str
    outputs: Optional[List[str]] = None
    inputs: Optional[List[str]] = None
    widget_inputs: Optional[Dict[str, Any]] = {}


# class Link(BaseModel):
#     """Represents a directed connection between two nodes in the graph.
#     - `source`: `str` Source node name
#     - `target`: `str` Target node name
#     - `source_handle`: `int` Output port on the source node.
#     - `target_handle`: `str` Input port on the target node.
#     - `key`: `str` key used to distinguish multiple links between the same nodes.
#     """

#     source: str
#     target: str
#     source_handle: int
#     target_handle: str
#     key: Optional[str] = "0"


# class Graph(BaseModel):
#     """Represents the entire graph structure, including nodes and links.
#     - `directed`: `bool`Whether the graph is directed.
#     - `multigraph`: `bool` Whether the graph allows multiple edges between the same nodes.
#     - `graph`: `dict` Additional metadata about the graph.
#     - `nodes`: `list[Node]` Node objects in the graph.
#     - `links`: `list[Link]` Connection objects between nodes.
#     """

#     directed: bool
#     multigraph: bool
#     graph: dict
#     nodes: List[Node]
#     links: List[Link]
