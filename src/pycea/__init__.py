from importlib.metadata import version

from . import datasets, pl, pp, tl, utils

__all__ = ["pl", "pp", "tl", "utils", "datasets"]

__version__ = version("pycea")
