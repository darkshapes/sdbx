from pydantic import BaseModel

from typing import Dict, List, Optional, Any

class Edge(BaseModel):
    source: str
    source_handle: int
    target_handle: str

class Node(BaseModel):
    id: str
    fname: str
    outputs: Optional[List[str]] = None
    inputs: Optional[List[str]] = None
    widget_inputs: Optional[Dict[str, Any]] = {}

class Link(BaseModel):
    source: str
    target: str
    source_handle: int
    target_handle: str
    key: Optional[str] = "0"

class Graph(BaseModel):
    directed: bool
    multigraph: bool
    graph: dict
    nodes: List[Node]
    links: List[Link]