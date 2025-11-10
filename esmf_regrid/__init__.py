"""
ESMF_regrid: A Python implementation of NCL's ESMF_regrid function.

This library provides regridding functionality using ESMPy, compatible with
the NCL ESMF_regrid interface.
"""

from .core import ESMF_regrid
from .options import RegridOptions

__version__ = "1.0.0"
__all__ = ["ESMF_regrid", "RegridOptions"]
