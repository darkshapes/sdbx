import os
import json
import logging
import tomllib
import argparse

import torch
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
    FP64 = "float64"
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8E4M3FN = "float8_e4m3fn"
    FP8E5M2 = "float8_e5m2"
    IN64 =  "int64"
    IN32 =  "int32"
    IN16 =  "int16"
    IN8 =  "int8"
    UN8 =  "uint8"
    NF4 = "nf4"

class TensorDataType:
    TYPE_T = Literal["F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I64", "I32", "I16", "I8", "U8", "nf4", "BOOL"]
    TYPE_R = Literal["fp64", "fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "i64", "i32", "i16", "i8", "u8", "nf4", "bool"]

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
    
    def get_path_tree(self, name, extension="", path_name=True, file_callback=lambda e: e, visited=None):
        p = self.get_path(name) if path_name else name
        visited = visited or set()

        def recurse(path):
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
                    entries.append({ **info, "children": recurse(fp) })
                elif entry.is_file(follow_symlinks=False) and (not extension or entry.name.endswith(extension)):
                    entries.append({ **info, **file_callback(fp) })
            return entries

        return recurse(p)

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
    
    def load_data(self, path):
        _, ext = os.path.splitext(path)
        loader, mode = (tomllib.load, "rb") if ext == ".toml" else (json.load, "r")
        with open(path, mode) as f:
            try:
                fd = loader(f)
            except (tomllib.TOMLDecodeError, json.decoder.JSONDecodeError) as e:
                raise SyntaxError(f"Couldn't read file {path}") from e
        return fd
    
    def get_default(self, name, prop):
        return self._defaults_dict[name][prop]

    @cached_property
    def _defaults_dict(self):
        d = {}
        glob_source = partial(glob, root_dir=config_source_location)
        for filename in glob_source("*.toml") + glob_source("*.json"):
            fp = os.path.join(config_source_location, filename)
            name, _ = os.path.splitext(filename)
            d[name] = self.load_data(fp)
        
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
        from sdbx.indexer import IndexManager
        return IndexManager()

    # @cached_property
    # def t2i_pipe(self):
    #     from sdbx.nodes.compute import T2IPipe
    #     return T2IPipe()

    @cached_property
    def node_tuner(self):
        from sdbx.nodes.tuner import NodeTuner
        return NodeTuner()
    
    def write_spec(self):
        import psutil
        from collections import defaultdict
        from platform import system
        spec = defaultdict(dict)
        spec["data"]["dynamo"]  = False if system().lower() == "windows" else True
        spec["data"]["devices"] = {}
        if torch.cuda.is_available():
            spec["data"]["devices"]["cuda"] = torch.cuda.mem_get_info()[1]
            spec["data"]["flash_attention"] = False #str(torch.backends.cuda.flash_sdp_enabled()).title()
            spec["data"]["allow_tf32"]      = False
            spec["data"]["xformers"]        = torch.backends.cuda.mem_efficient_sdp_enabled()
            if "True" in [spec["data"].get("xformers"), spec["data"].get("flash_attention")]:
                spec["data"]["enable_attention_slicing"] = False
        if torch.backends.mps.is_available() & self.device.backends.mps.is_built():
            spec["data"]["devices"]["mps"] = torch.mps.driver_allocated_memory()
            try: 
                import flash_attn
            except: 
                spec["data"]["flash_attention"]          = False
                spec["data"]["enable_attention_slicing"] = True
            else:
                spec["data"]["flash_attention"] = True  # hope for the best that user set this up
            #set USE_FLASH_ATTENTION=1 in console
            # ? https                       : //pytorch.org/docs/master/notes/mps.html
            # ? memory_fraction = 0.5  https: //iifx.dev/docs/pytorch/generated/torch.mps.set_per_process_memory_fraction
            # ? torch.mps.set_per_process_memory_fraction(memory_fraction)
        if torch.xpu.is_available():
            # todo: code for xpu total memory, possibly code for mkl
            """ spec["data"]["devices"]["xps"] = ram"""
        spec["data"]["devices"]["cpu"] = psutil.virtual_memory().total # set all floats = fp32
        spec_file = os.path.join(config_source_location, "spec.json")
        if os.path.exists(spec_file):
            try:
                os.remove(spec_file)
            except FileNotFoundError as error_log:
                logging.debug(f"'Spec file absent at write time: {spec_file}.'{error_log}", exc_info=True)
        if spec:
            try:
                with open(spec_file, "w+", encoding="utf8") as file_out:
                    """ try opening file"""
            except Exception as error_log:
                logging.debug(f"Error writing spec file '{spec_file}': {error_log}", exc_info=True)
            else:
                with open(spec_file, "w+", encoding="utf8") as file_out:
                    json.dump(spec, file_out, ensure_ascii=False, indent=4, sort_keys=False)
        else:
            logging.debug("No data to write to spec file.", exc_info=True)
        #return data


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