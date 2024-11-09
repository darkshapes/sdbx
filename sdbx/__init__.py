from importlib.metadata import version, PackageNotFoundError # setuptools-scm versioning
try:
    __version__ = version("sdbx")
except PackageNotFoundError:
    # package is not installed
    pass

import logging
logger = logging.getLogger('uvicorn.error')

from .config import config, source
if "dev" in __version__: config.development = True

from .server import create_app as app