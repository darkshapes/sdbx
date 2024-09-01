import os
import sys
import site
import runpy
import logging
import pkgutil
import tomllib
import importlib
import sysconfig
import subprocess
from typing import Callable, Dict, List
from inspect import getmembers, isfunction
from importlib.metadata import distribution
from functools import cached_property, cache

import virtualenv
from dulwich import porcelain

class NodeManager:
	def __init__(self, path, nodes_path, env_name=".node_env"):
		self.path = path
		self.nodes_path = nodes_path
		
		with open(path, 'rb') as file:
			self.node_modules = tomllib.load(file)["nodes"]
		
		self.initialize_environment(env_name)

		# self.node_module_names = ["sdbx.nodes.base"] + [self.validate_node_installed(n, u) for n, u in self.node_modules.items()]
		self.node_module_names = ["sdbx.nodes.base"] # NOTE: ignoring comfyextras only for the time being
	
	def initialize_environment(self, env_name=".node_env"):		
		# Create environment if it doesn't exist
		self.env_path = os.path.join(os.path.dirname(self.path), env_name)
		if not os.path.exists(self.env_path):
			logging.info("Creating node environment...")
			virtualenv.cli_run([str(self.env_path)])
		
		self.env_vars = {"base": self.env_path, "platbase": self.env_path}
		self.env_bin = sysconfig.get_path("scripts", vars=self.env_vars)
		self.env_python = os.path.join(self.env_bin, "python")
		self.env_package_path = sysconfig.get_path("platlib", vars=self.env_vars)

		runpy.run_path(os.path.join(self.env_bin, "activate_this.py")) # Add node env packages to our current environment

		parent_package_path = site.getsitepackages()[0]
		parent_link_path = os.path.join(self.env_package_path, "parent.pth")
		if not os.path.exists(parent_link_path):
			with open(parent_link_path, 'w') as f:
				f.write(parent_package_path)

		self.env_pip = os.path.join(self.env_bin, "pip")
		self.env_packages = [p.metadata['Name'] for p in importlib.metadata.distributions()]
	
	def validate_node_installed(self, node_module, url):
		node_path = os.path.join(self.nodes_path, os.path.normpath(node_module))
		node_name = os.path.basename(node_module)

		if not os.path.exists(node_path): # check all manifest nodes downloaded
			logging.info(f"Downloading {node_module}...")
			os.makedirs(node_path, exist_ok=True)
			porcelain.clone(url, node_path)
		
		if node_name not in self.env_packages: # check all downloaded nodes installed
			logging.info(f"Installing {node_module}...")
			subprocess.check_call([self.env_pip, "install", "-e", node_path])
		
		return node_name
	
	@cached_property
	def nodes(self) -> List[Callable]:
		def get_nodes(module_name): # Get all functions marked with @node decorator
			module = importlib.import_module(module_name)
			functions = [fn for _, fn in getmembers(module, isfunction) if hasattr(fn, 'info')]

			# Recursively find functions in submodules
			for finder, name, ispkg in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
				submodule = importlib.import_module(name)
				functions.extend([fn for _, fn in getmembers(submodule, isfunction) if hasattr(fn, 'info')])

			return functions

		return [fn for module_name in self.node_module_names for fn in get_nodes(module_name)]
	
	@cached_property
	def node_info(self) -> Dict[str, dict]:
		return { node.info.name: node.info.dict() for node in self.nodes }
	
	@cached_property
	def registry(self) -> Dict[str, Callable]:
		return { node.__name__: cache(node) for node in self.nodes }
