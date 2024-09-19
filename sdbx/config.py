import os
import json
import logging
import tomllib
import argparse

from enum import Enum
from glob import glob
from pathlib import Path
from typing import Tuple, Type, Union, Literal, List
from functools import cache, cached_property, partial, total_ordering

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

source = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_source_location = os.path.join(source, "config")

@cache
def get_config_location():
    from platform import system

    filename = "config.toml"

    return {
        'windows': os.path.join(os.environ.get('LOCALAPPDATA', os.path.join(os.path.expanduser('~'), 'AppData', 'Local')), 'Shadowbox', filename),
        'darwin': os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'Shadowbox', filename),
        'linux': os.path.join(os.path.expanduser('~'), '.config', 'shadowbox', filename),
    }[system().lower()]

class LatentPreviewMethod(str, Enum):
    NONE = "none"
    AUTO = "auto"
    LATENT2RASTER = "latent2raster"
    TAESD = "taesd"

@total_ordering
class VRAM(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    NONE = "none"

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            order = [VRAM.NONE, VRAM.LOW, VRAM.NORMAL, VRAM.HIGH]
            return order.index(self) < order.index(other)
        return NotImplemented

class Precision(str, Enum):
    MIXED = "mixed"
    F64 = "float64"
    F32 = "float32"
    F16 = "float16"
    BF16 = "bfloat16"
    FP8E4M3FN = "float8_e4m3fn"
    FP8E5M2 = "float8_e5m2"

# class TensorType:
#     DTYPE_T = Literal["F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"]

# class TensorData:
#     dtype: DTYPE_T
#     shape: List[int]
#     data_offsets: Tuple[int, int]
#     parameter_count: int = field(init=False)

#     def __post_init__(self) -> None:
#         # Taken from https://stackoverflow.com/a/13840436
#         try:
#             self.parameter_count = functools.reduce(operator.mul, self.shape)
#         except TypeError:
#             self.parameter_count = 1  # scalar value has no shape

class ConfigModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda s: s.replace('_', '-')
    )


class ExtensionsConfig(ConfigModel):
    disable: Union[bool, Literal['clients', 'nodes']] = False

class LocationConfig(ConfigModel):
    clients: str = "clients"
    nodes: str = "nodes"
    flows: str = "flows"
    input: str = "input"
    output: str = "output"
    models: str = "models"

class WebConfig(ConfigModel):
    listen: str = "127.0.0.1"
    port: int = 8188
    reload: bool = True
    reload_include: List[str] = ["*.toml", "*.json", "models/**/*"]
    external_address: str = "localhost"
    max_upload_size: int = 100
    auto_launch: bool = True
    known_models: bool = True
    preview_mode: LatentPreviewMethod = LatentPreviewMethod.AUTO

class ComputationalConfig(ConfigModel):
    deterministic: bool = False

class MemoryConfig(ConfigModel):
    vram: VRAM = VRAM.NORMAL
    smart_memory: bool = True

MixedPrecision = Union[Literal[Precision.MIXED, Precision.F32, Precision.F16, Precision.BF16]]
EncoderPrecision = Union[Literal[Precision.F32, Precision.F16, Precision.BF16, Precision.FP8E4M3FN, Precision.FP8E5M2]]

class PrecisionConfig(ConfigModel):
    fp: MixedPrecision = Precision.MIXED
    unet: EncoderPrecision = Precision.F32
    vae: MixedPrecision = Precision.MIXED
    text_encoder: EncoderPrecision = Precision.F16

class DistributedConfig(ConfigModel):
    role: Literal[False, Literal['worker', 'frontend']] = False
    name: str = "shadowbox"
    connection_uri: str = "amqp://guest:guest@127.0.0.1"

class OrganizationConfig(ConfigModel):
    channels_first: bool = False

class Config(BaseSettings):
    """
    Configuration options parsed from config.toml.
    """
    model_config = SettingsConfigDict(env_prefix="SDBX_")

    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig)
    location: LocationConfig = Field(default_factory=LocationConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    computational: ComputationalConfig = Field(default_factory=ComputationalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    organization: OrganizationConfig = Field(default_factory=OrganizationConfig)

    development: bool = False

    def __new__(cls, path: str, *args, **kwargs):
        cls.model_config['toml_file'] = path
        cls.path = os.path.dirname(path)
        return super().__new__(cls)

    def model_post_init(self, __context):
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
        return (TomlConfigSettingsSource(settings_cls),)
        
    def generate_new_config(self):
        logging.info(f"Creating config directory at {self.path}...")

        os.makedirs(self.path, exist_ok=True)

        for subdir in self._path_dict.values():
            print(os.path.join(self.path, subdir))
            os.makedirs(os.path.join(self.path, subdir), exist_ok=True)
        
        from shutil import copytree
        copytree(os.path.join(config_source_location, "user"), self.path, dirs_exist_ok=True)

    def rewrite(self, key, value):
        # rewrites the config.toml key to value
        pass
    
    def get_path(self, name):
        return self._path_dict[name]
    
    def get_path_contents(self, name, extension="", path_name=True, base_name=False):  # List contents of a directory
        p = self.get_path(name) if path_name else name

        format_path = lambda p, g: os.path.join(p, g)

        if base_name:  # TODO: should the client receive full paths?
            format_path = lambda p, g: os.path.basename(g)  # Only return base name

        return [format_path(p, g) for g in glob(f"**.{extension}", root_dir=p, recursive=True)]
    
    def get_path_tree(self, name, extension="", path_name=True, file_callback=(lambda e: e), _searching=None):
        p = self.get_path(name) if path_name else name
        _searching = _searching or ""
        tree = []

        recurse_tree = partial(self.get_path_tree, name=p, path_name=False, file_callback=file_callback)

        with os.scandir(os.path.join(p, _searching)) as it:
            for entry in it:
                if not entry.is_dir() and not entry.name.endswith(extension):
                    continue
                current = os.path.join(_searching, entry.name)
                full = os.path.join(p, current)
                tree.append({
                    "id": current,
                    "name": entry.name,
                    **(
                        file_callback(full) if entry.is_file() else  # If it's a file, load contents as graph
                        { "children": recurse_tree(_searching=current) }  # If it's a directory, recurse
                    )
                })
        
        return tree

    @cached_property
    def _path_dict(self):
        root = {
            n: os.path.join(self.path, p) for n, p in dict(self.location).items() # see self.location for details
        }

        for n, p in dict(self.location).items():
            if ".." in p:
                raise Exception("Cannot set location outside of config path.")

        models = {f"models.{name}": os.path.join(root["models"], name) for name in self.get_default("directories", "models")}

        return {**root, **models}
    
    def get_default(self, name, prop):
        return self._defaults_dict[name][prop]
    
    @cached_property
    def _defaults_dict(self):
        d = {}
        glob_source = partial(glob, root_dir=config_source_location)

        for filename in glob_source("*.toml") + glob_source("*.json"):
            filepath = Path(os.path.join(config_source_location, filename))
            ext = filepath.suffix
            loader, mode = (tomllib.load, "rb") if ext == ".toml" else (json.load, "r")
            with open(filepath, mode) as f:
                try:
                    fd = loader(f)
                except (tomllib.TOMLDecodeError, json.decoder.JSONDecodeError) as e:
                    raise SyntaxError(f"Couldn't read file {filepath}") from e

            name = filepath.stem
            d[name] = fd
        
        return d
    
    @cached_property
    def extension_data(self):
        with open(os.path.join(self.path, "extensions.toml"), "rb") as f:
            return tomllib.load(f)

    @cached_property
    def node_manager(self):
        from sdbx.nodes.manager import NodeManager # we must import this here lest we summon the dreaded ouroboros
        return NodeManager(self.extension_data, nodes_path=self.get_path("nodes"))

    @cached_property
    def client_manager(self):
        from sdbx.clients.manager import ClientManager
        return ClientManager(self.extension_data, clients_path=self.get_path("clients"))

    @cached_property
    def executor(self):
        from sdbx.executor import Executor
        return Executor(self.node_manager)
    
    @cached_property
    def model_indexer(self):
        from sdbx.indexer import ModelIndexer
        return ModelIndexer()


def parse() -> Config:
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('-c', '--config', type=str, default=get_config_location(), help='Location of the config file.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('-s', '--silent', action='store_true', help='Silence all print to stdout.')
    parser.add_argument('-d', '--daemon', action='store_true', help='Run in daemon mode (not associated with tty).')
    parser.add_argument('-h', '--help', action='help', help='See config.toml for more configuration options.')
    # parser.add_argument('--setup', action='store_true', help='Setup and exit.')

    args = parser.parse_args()

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    if args.silent:
        level = logging.ERROR
    
    logging.basicConfig(encoding='utf-8', level=level)
    
    return Config(path=args.config)

config = parse()