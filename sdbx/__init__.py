from importlib.metadata import version, PackageNotFoundError

# setuptools-scm versioning
try:
    __version__ = version("sdbx")
except PackageNotFoundError:
    # package is not installed
    pass

from .config import config
from .server import create_app

app = create_app()