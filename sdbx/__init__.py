from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sdbx")
except PackageNotFoundError:
    # package is not installed
    pass