import importlib
import logging
import os
import pkgutil
import runpy
import site
import sysconfig
from functools import cached_property
from inspect import getmembers, isfunction
from typing import Callable, Dict, List

import virtualenv

from sdbx.config import ExtensionRegistry
from sdbx.nodes.helpers import cache


class NodeManager:
    """Ensure node extensions are installed in a venv and available for use."""

    def __init__(self, node_modules: ExtensionRegistry, nodes_path: str, env_name: str = ".node_env") -> None:
        """Create an instance of NodeManager\n
        :param node_modules: Auxillary node extensions installed by user
        :param nodes_path: The install location for node extensions
        :param env_name: Location for the Python venv to constrain node dependencies in, defaults to ".node_env"
        """
        self.node_modules = node_modules
        self.nodes_path = nodes_path

        self.initialize_environment(env_name)

        # self.node_module_names = ["sdbx.nodes.base"] + [self.validate_node_installed(n, u) for n, u in self.node_modules.items()]
        self.node_module_names = ["sdbx.nodes.base"]  # NOTE: ignoring comfyextras only for the time being

    def initialize_environment(self, env_name: str = ".node_env") -> None:
        """Access or create a Python venv for separating node dependencies from main dependencies\n
        :param env_name: Location for the Python venv to constrain node dependencies in, defaults to ".node_env"
        """
        # Create environment if it doesn't exist
        self.env_path = os.path.join(os.path.dirname(self.nodes_path), env_name)
        if not os.path.exists(self.env_path):
            logging.info("Creating node environment...")
            virtualenv.cli_run([str(self.env_path)])

        self.env_vars = {"base": self.env_path, "platbase": self.env_path}
        self.env_bin = sysconfig.get_path("scripts", vars=self.env_vars)
        self.env_python = os.path.join(self.env_bin, "python")
        self.env_package_path = sysconfig.get_path("platlib", vars=self.env_vars)

        runpy.run_path(os.path.join(self.env_bin, "activate_this.py"))  # Add node env packages to our current environment

        parent_package_path = site.getsitepackages()[0]
        parent_link_path = os.path.join(self.env_package_path, "parent.pth")
        if not os.path.exists(parent_link_path):
            with open(parent_link_path, "w", encoding="utf-8") as f:
                f.write(parent_package_path)

        self.env_pip = os.path.join(self.env_bin, "pip")
        self.env_packages = [p.metadata["Name"] for p in importlib.metadata.distributions()]

    # The convenience of pulling nodes from repos has proven very insecure for other projects.
    # We should provide a robust solution when ready.

    # def validate_node_installed(self, node_module: str, url: str):
    # 	"""Pull nodes from a remote repo\n
    # 	:param node_module" The node to access
    # 	:param url: The url to access it from
    # 	:return: The name of the node
    # 	"""

    #     node_path = os.path.join(self.nodes_path, os.path.normpath(node_module))
    #     node_name = os.path.basename(node_module)

    #     if not os.path.exists(node_path):  # check all manifest nodes downloaded
    #          logging.info(f"Downloading {node_module}...")
    #          os.makedirs(node_path, exist_ok=True)
    #          porcelain.clone(url, node_path)

    #     if node_name not in self.env_packages:  # check all downloaded nodes installed
    #          logging.info(f"Installing {node_module}...")
    #          subprocess.check_call([self.env_pip, "install", "-e", node_path])

    #     return node_name

    @cached_property
    def nodes(self) -> List[Callable]:
        """Retrieve all callable nodes
        :return: A list of nodes
        """

        def get_nodes(module_name: str) -> Callable:
            """Get all functions marked with @node decorator\n
            Nodes have the attribute `info` to distinguish them from normal functions
            :param module_name: The file name of the node module to search within
            :return: The functions matching the search criteria
            """
            module = importlib.import_module(module_name)
            functions = [fn for _, fn in getmembers(module, isfunction) if hasattr(fn, "info")]

            # Recursively find functions in submodules
            for _finder, name, _ispkg in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
                submodule = importlib.import_module(name)
                functions.extend([fn for _, fn in getmembers(submodule, isfunction) if hasattr(fn, "info")])

            return functions

        return [fn for module_name in self.node_module_names for fn in get_nodes(module_name)]

    @cached_property
    def node_info(self) -> Dict[str, dict]:
        """Mapping node names to metadata\n
        :return: Dictionary of node names -> metadata
        """
        return {node.info.name: node.info.dict() for node in self.nodes}

    @cached_property
    def registry(self) -> Dict[str, Callable]:
        """Each node’s name is mapped to its callable function
        :return:  Dictionary of node names -> function
        """
        return {node.__name__: cache(node) for node in self.nodes}
