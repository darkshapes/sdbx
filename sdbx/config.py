import os
import enum
import glob
import shutil
import logging
import tomllib
import argparse
import platform

from pathlib import Path
from typing import Tuple, Type, Union, Literal
from functools import total_ordering, cached_property

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

config_source_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config")

def get_config_location():
    filename = "config.toml"

    return {
        'windows': os.path.join(os.environ.get('LOCALAPPDATA', os.path.join(os.path.expanduser('~'), 'AppData', 'Local')), 'Shadowbox', filename),
        'darwin': os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', 'Shadowbox', filename),
        'linux': os.path.join(os.path.expanduser('~'), '.config', 'shadowbox', filename),
    }[platform.system().lower()]


class LatentPreviewMethod(str, enum.Enum):
    NONE = "none"
    AUTO = "auto"
    LATENT2RGB = "latent2rgb"
    TAESD = "taesd"

@total_ordering
class VRAM(str, enum.Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    NONE = "none"

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            order = [VRAM.NONE, VRAM.LOW, VRAM.NORMAL, VRAM.HIGH]
            return order.index(self) < order.index(other)
        return NotImplemented

class Precision(str, enum.Enum):
    MIXED = "mixed"
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8E4M3FN = "float8_e4m3fn"
    FP8E5M2 = "float8_e5m2"


class ConfigModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=lambda s: s.replace('_', '-')
    )


class ExtensionsConfig(ConfigModel):
    disable: Union[bool, Literal['clients', 'nodes']] = False

class LocationConfig(ConfigModel):
    clients: str = "clients"
    nodes: str = "nodes"
    input: str = "input"
    output: str = "output"
    models: str = "models"
    workflows: str = "workflows"

class WebConfig(ConfigModel):
    listen: str = "127.0.0.1"
    port: int = 8188
    external_address: str = "localhost"
    enable_cors_header: Union[bool, str] = False
    max_upload_size: int = 100
    auto_launch: bool = True
    known_models: bool = True
    preview_mode: LatentPreviewMethod = LatentPreviewMethod.AUTO

class ComputationalConfig(ConfigModel):
    gpu_only: bool = False
    cuda_device: int = 0
    cuda_malloc: bool = True
    cpu_only: bool = False
    cpu_vae: bool = True
    directml: Union[bool, int] = False
    ipex_optimize: bool = False
    xformers: bool = True
    cross_attention: Literal['split', 'quad', 'torch'] = "torch"
    upcast_attention: Union[bool, Literal['force']] = True
    deterministic: bool = False

class MemoryConfig(ConfigModel):
    vram: VRAM = VRAM.NORMAL
    smart_memory: bool = True

MixedPrecision = Union[Literal[Precision.MIXED, Precision.FP32, Precision.FP16, Precision.BF16]]
EncoderPrecision = Union[Literal[Precision.FP32, Precision.FP16, Precision.BF16, Precision.FP8E4M3FN, Precision.FP8E5M2]]

class PrecisionConfig(ConfigModel):
    fp: MixedPrecision = Precision.MIXED
    unet: EncoderPrecision = Precision.FP32
    vae: MixedPrecision = Precision.MIXED
    text_encoder: EncoderPrecision = Precision.FP16

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

    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig)
    location: LocationConfig = Field(default_factory=LocationConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    computational: ComputationalConfig = Field(default_factory=ComputationalConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)
    organization: OrganizationConfig = Field(default_factory=OrganizationConfig)

    development: bool = False

    def __init__(self, path: str):
        if not isinstance(path, str):
            raise TypeError("Config path must be a string")

        Config.path = path if os.path.exists(path) else os.path.join(config_source_location, "user")
        super().__init__()
        Config.path = os.path.dirname(path)
        
        if not os.path.exists(path):
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
        return (TomlConfigSettingsSource(settings_cls, settings_cls.path),)
        
    def generate_new_config(self):
        logging.info(f"Creating config directory at {self.path}...")

        os.makedirs(self.path, exist_ok=True)

        for subdir in self._path_dict.values():
            print(os.path.join(self.path, subdir))
            os.makedirs(os.path.join(self.path, subdir), exist_ok=True)
        
        shutil.copytree(os.path.join(config_source_location, "user"), self.path, dirs_exist_ok=True)

    def rewrite(self, key, value):
        # rewrites the config.toml key to value
        pass
    
    def get_path(self, name):
        return self._path_dict[name]
    
    def get_path_contents(self, name, extension=""):
        p = self.get_path(name)
        return [os.path.join(p, g) for g in glob.glob(f"**.{extension}", root_dir=p, recursive=True)]

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

        for filename in glob.glob("*.toml", root_dir=config_source_location):
            filepath = Path(os.path.join(config_source_location, filename))
            with open(filepath, "rb") as f:
                fd = tomllib.load(f)
            name = filepath.stem
            d[name] = fd
        
        return d
    
    @cached_property
    def extensions_path(self):
        return os.path.join(self.path, "extensions.toml")

    @cached_property
    def node_manager(self):
        from sdbx.nodes.manager import NodeManager # we must import this here lest we summon the dreaded ouroboros
        return NodeManager(self.extensions_path, nodes_path=self.get_path("nodes"))

    @cached_property
    def client_manager(self):
        from sdbx.clients.manager import ClientManager
        return ClientManager(self.extensions_path, clients_path=self.get_path("clients"))

    @cached_property
    def executor(self):
        from sdbx.executor import Executor
        return Executor(self.node_manager)


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
    
    return Config(args.config)

config = parse()