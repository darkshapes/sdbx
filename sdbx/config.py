import argparse
import json
import logging
import os
import sys
import tomllib
from io import TextIOWrapper

# from enum import Enum
from functools import cache, cached_property, partial  # , total_ordering
from glob import glob

# from pathlib import Path
from typing import Callable, ClassVar, Generator, List, Literal, Set, Tuple, Type, Union

import torch  # noqa: F811
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# pylint: disable=unnecessary-lambda, unnecessary-lambda-assignment

source = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_source_location = os.path.join(source, "config")


@cache
def get_config_location() -> dict[str]:
    """Return user the config folder for each platform type
    :return: A dictionary keyed by OS name
    """
    from platform import system

    filename = "config.toml"

    return {
        "windows": os.path.join(os.environ.get("LOCALAPPDATA", os.path.join(os.path.expanduser("~"), "AppData", "Local")), "Shadowbox", filename),
        "darwin": os.path.join(os.path.expanduser("~"), "Library", "Application Support", "Shadowbox", filename),
        "linux": os.path.join(os.path.expanduser("~"), ".config", "shadowbox", filename),
    }[system().lower()]


class ConfigModel(BaseModel):
    """Config file system"""

    model_config = ConfigDict(alias_generator=lambda s: s.replace("_", "-"))


class ExtensionsConfig(ConfigModel):
    """Eextension setting toggle"""

    disable: Union[bool, Literal["clients", "nodes"]] = False


class LocationConfig(ConfigModel):
    """Default directory paths"""

    clients: str = "clients"
    nodes: str = "nodes"
    flows: str = "flows"
    input: str = "input"
    output: str = "output"
    models: str = "models"


class WebConfig(ConfigModel):
    """Server configuration setup"""

    listen: str = "127.0.0.1"
    port: int = 8188
    reload: bool = True
    reload_include: List[str] = ["*.toml", "*.json", "models/**/*"]
    external_address: str = "localhost"
    max_upload_size: int = 100
    auto_launch: bool = True
    known_models: bool = True


class ComputationalConfig(ConfigModel):
    """Toggle deterministic seeding (CUDA)"""

    deterministic: bool = False


class MemoryConfig(ConfigModel):
    """Toggle system spec gathering on start"""

    system_profiling: bool = True


class Config(BaseSettings):
    """Configuration options parsed from config.toml."""

    model_config = SettingsConfigDict(env_prefix="SDBX_")

    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig)
    location: LocationConfig = Field(default_factory=LocationConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    computational: ComputationalConfig = Field(default_factory=ComputationalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    development: bool = False  # Whether this is a development build

    def __new__(cls, path: str, *args, **kwargs) -> Callable:
        """Recreate the main config file"""
        cls.model_config["toml_file"] = path
        cls.path = os.path.dirname(path)
        return super().__new__(cls)

    def model_post_init(self, __context) -> None:  # pylint: disable=arguments-differ
        """Trigger new config build if folders do not exist"""
        if not os.path.exists(self.path):
            self.generate_new_config()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Settings locations for base settings, environment variables, password secrets, etc"""
        return (TomlConfigSettingsSource(settings_cls),)

    def generate_new_config(self) -> None:
        """Rebuild the config files"""
        logging.info("%s", f"Creating config directory at {self.path}...")

        os.makedirs(self.path, exist_ok=True)

        for subdir in self._path_dict.values():
            print(os.path.join(self.path, subdir))
            os.makedirs(os.path.join(self.path, subdir), exist_ok=True)

        from shutil import copytree

        copytree(os.path.join(config_source_location, "user"), self.path, dirs_exist_ok=True)

    def rewrite(self, key, value) -> None:
        """Rewrites the config.toml key to value"""
        # Not Implemented

    def get_path(self, name) -> dict[str]:
        """Helper function for path"""
        return self._path_dict[name]

    def get_path_contents(self, name, extension="", path_name=True, base_name=False) -> Generator:
        """List contents of a directory"""
        p = self.get_path(name) if path_name else name

        format_path = lambda p, g: os.path.join(p, g)

        if base_name:  # TODO: should the client receive full paths?
            format_path = lambda p, g: os.path.basename(g)  # Only return base name

        return [format_path(p, g) for g in glob(f"**.{extension}", root_dir=p, recursive=True)]

    def get_path_tree(
        self,
        name: str,
        extension: str = "",
        path_name: bool = True,
        file_callback: Callable = lambda e: e,
        visited: set[str] = None,
    ) -> Callable | List[str]:
        """Trace file system, marking locations as visited along the way

        :param name: Location to trace
        :param extension: Suffix type to *exclude*, defaults to ""
        :param path_name: Whether `name` is a cached location, otherwise access like normal, defaults to True
        :param file_callback: Additional function to perform on each file, defaults to lambda e:e (passthrough)
        :param visited: Canonical, non-symbolic link of current progress through folders, defaults to None
        :return: All files and folder paths within `name`
        """
        p = self.get_path(name) if path_name else name
        visited = visited or set()

        def recurse(path) -> set[str]:
            entries = []
            for entry in os.scandir(path):
                fp = os.path.join(path, entry.name)
                rp = os.path.realpath(fp)
                if rp in visited:
                    continue
                visited.add(rp)
                info = {
                    "id": fp,
                    "name": entry.name,
                }
                if entry.is_dir(follow_symlinks=True):
                    entries.append({**info, "children": recurse(fp)})
                elif entry.is_file(follow_symlinks=False) and (not extension or entry.name.endswith(extension)):
                    entries.append({**info, **file_callback(fp)})
            return entries

        return recurse(p)

    @cached_property
    def _path_dict(self) -> set[str]:
        """Cache a map of names to file paths, generate paths for model sub directories"""
        root = {n: os.path.join(self.path, p) for n, p in dict(self.location).items()}  # see self.location for details

        for n, p in dict(self.location).items():
            if ".." in p:
                raise ValueError("Cannot set location outside of config path.")

        models = {f"models.{name}": os.path.join(root["models"], name) for name in self.get_default("directories", "models")}

        return {**root, **models}

    def load_data(self, path: str) -> TextIOWrapper:
        """Load TOML/JSON file contents

        :param path: Path to file
        :raises SyntaxError: File couldn't be read
        :return: File contents
        """
        _, ext = os.path.splitext(path)
        loader, mode = (tomllib.load, "rb") if ext == ".toml" else (json.load, "r")
        with open(path, mode) as f:
            try:
                fd = loader(f)
            except (tomllib.TOMLDecodeError, json.decoder.JSONDecodeError) as e:
                raise SyntaxError(f"Couldn't read file {path}") from e
        return fd

    def get_default(self, name: str | int, prop: str | int) -> dict[str | bool | int | float]:
        """Retrieve configuration from file

        :param name: The prefix of a configuration file
        :param prop: The setting from the file to retrieve
        :return: A mapping of `name` to its contents
        """
        return self._defaults_dict[name][prop]

    @cached_property
    def _defaults_dict(self) -> dict[str]:
        """Map TOML/JSON files to their contents"""
        d = {}
        glob_source = partial(glob, root_dir=config_source_location)
        for filename in glob_source("*.toml") + glob_source("*.json"):
            fp = os.path.join(config_source_location, filename)
            name, _ = os.path.splitext(filename)
            d[name] = self.load_data(fp)

        return d

    @cached_property
    def extension_data(self) -> TextIOWrapper:
        """Additional extensions to load with the system"""
        with open(os.path.join(self.path, "extensions.toml"), "rb") as f:
            return tomllib.load(f)

    @cached_property
    def node_manager(self) -> Callable:
        """**`B̴̨̒e̷w̷͇̃ȁ̵͈r̸͔͛ę̵͂ ̷̫̚t̵̻̐h̶̜͒ȩ̸̋ ̵̪̄ő̷̦ù̵̥r̷͇̂o̷̫͑b̷̲͒ò̷̫r̴̢͒ô̵͍s̵̩̈́`**"""
        from sdbx.nodes.manager import NodeManager  # we must import this here lest we summon the dreaded ouroboros

        return NodeManager(self.extension_data, nodes_path=self.get_path("nodes"))

    @cached_property
    def client_manager(self) -> Callable:
        """Client to use for the system"""
        from sdbx.clients.manager import ClientManager

        return ClientManager(self.extension_data, clients_path=self.get_path("clients"))

    @cached_property
    def executor(self) -> Callable:
        """Puts nodes to work"""
        from sdbx.executor import Executor

        return Executor(self.node_manager)


def parse(testing: bool = False) -> Config:
    if "pytest" in sys.modules:
        testing = True

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("-c", "--config", type=str, default=get_config_location(), help="Location of the config file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-s", "--silent", action="store_true", help="Silence all print to stdout.")
    parser.add_argument("-d", "--daemon", action="store_true", help="Run in daemon mode (not associated with tty).")
    parser.add_argument("-h", "--help", action="help", help="See config.toml for more configuration options.")
    # parser.add_argument('--setup', action='store_true', help='Setup and exit.')

    args = parser.parse_args() if not testing else parser.parse_args([])

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    if args.silent:
        level = logging.ERROR

    logging.basicConfig(encoding="utf-8", level=level)

    return Config(path=args.config)


config = parse(testing=hasattr(sys, "_called_from_test"))
