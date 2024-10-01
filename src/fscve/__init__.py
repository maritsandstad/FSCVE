"""
Init
"""

from . import _version
from .fscve import FSCVE  # noqa: F401

__version__ = _version.get_versions()["version"]
