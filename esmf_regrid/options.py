"""
Options class for ESMF_regrid configuration.

This module defines the RegridOptions class which holds all configuration
parameters for the regridding operation, matching the NCL ESMF_regrid interface.
"""

import numpy as np
from typing import Optional, Union, List


class RegridOptions:
    """
    Configuration options for ESMF_regrid.

    This class encapsulates all the options that can be passed to ESMF_regrid,
    similar to the NCL opt variable.
    """

    def __init__(self):
        # File names
        self.SrcFileName: str = "source_grid_file.nc"
        self.DstFileName: str = "destination_grid_file.nc"
        self.WgtFileName: str = "weights_file.nc"

        # Interpolation method
        # Options: "bilinear", "patch", "conserve", "neareststod", "nearestdtos"
        self.InterpMethod: str = "bilinear"

        # Pole handling for periodic grids
        # Options: "none", "all", "teeth"
        self.Pole: str = "all"

        # Grid coordinates (source)
        self.SrcGridLat: Optional[np.ndarray] = None
        self.SrcGridLon: Optional[np.ndarray] = None
        self.SrcGridCornerLat: Optional[np.ndarray] = None
        self.SrcGridCornerLon: Optional[np.ndarray] = None

        # Grid coordinates (destination)
        self.DstGridLat: Optional[np.ndarray] = None
        self.DstGridLon: Optional[np.ndarray] = None
        self.DstGridCornerLat: Optional[np.ndarray] = None
        self.DstGridCornerLon: Optional[np.ndarray] = None

        # Grid types (e.g., "1x1", "0.25deg", etc.)
        self.SrcGridType: Optional[str] = None
        self.DstGridType: Optional[str] = None

        # Masking
        self.SrcMask2D: Optional[np.ndarray] = None
        self.DstMask2D: Optional[np.ndarray] = None
        self.SrcGridMask: Optional[np.ndarray] = None
        self.DstGridMask: Optional[np.ndarray] = None

        # Missing values
        self.SrcMissingValue: Optional[float] = None
        self.DstMissingValue: Optional[float] = None

        # Regional grids
        self.SrcRegional: bool = False
        self.DstRegional: bool = False

        # Tripoler grids
        self.SrcTripoler: bool = False
        self.DstTripoler: bool = False

        # Grid corners
        self.SrcGridCorners: Optional[int] = None
        self.DstGridCorners: Optional[int] = None

        # Input dimension names
        self.SrcInputFileName: Optional[str] = None
        self.DstInputFileName: Optional[str] = None
        self.SrcLatName: Optional[str] = None
        self.SrcLonName: Optional[str] = None
        self.DstLatName: Optional[str] = None
        self.DstLonName: Optional[str] = None

        # Output options
        self.CopyVarCoords: bool = True
        self.CopyVarAtts: bool = True
        self.ReturnDouble: bool = False

        # File options
        self.ForceOverwrite: bool = False
        self.RemoveSrcFile: bool = False
        self.RemoveDstFile: bool = False
        self.RemoveWgtFile: bool = False
        self.NetCDFType: str = "netcdf4"  # "netcdf4" or "netcdf4classic"

        # Debug options
        self.PrintTimings: bool = False
        self.Debug: bool = False

        # Advanced ESMF options
        self.IgnoreUnmappedPoints: bool = False
        self.LineType: str = "GREAT_CIRCLE"  # "GREAT_CIRCLE" or "CART"
        self.NormType: str = "FRACAREA"  # For conservative regridding
        self.ExtrapMethod: Optional[str] = None  # "neareststod", "nearestidavg", "creep"
        self.ExtrapNumSrcPnts: int = 8
        self.ExtrapDistExponent: float = 2.0
        self.ExtrapNumLevels: int = 0

        # Unstructured grid options
        self.SrcGridMeshElements: Optional[np.ndarray] = None
        self.DstGridMeshElements: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, opt_dict: dict) -> 'RegridOptions':
        """
        Create a RegridOptions object from a dictionary.

        Parameters
        ----------
        opt_dict : dict
            Dictionary of options

        Returns
        -------
        RegridOptions
            Configured options object
        """
        options = cls()
        for key, value in opt_dict.items():
            if hasattr(options, key):
                setattr(options, key, value)
            else:
                raise ValueError(f"Unknown option: {key}")
        return options

    def validate(self):
        """
        Validate the options.

        Raises
        ------
        ValueError
            If options are invalid
        """
        valid_methods = ["bilinear", "patch", "conserve", "neareststod", "nearestdtos"]
        if self.InterpMethod not in valid_methods:
            raise ValueError(
                f"Invalid InterpMethod: {self.InterpMethod}. "
                f"Must be one of {valid_methods}"
            )

        valid_poles = ["none", "all", "teeth"]
        if self.Pole not in valid_poles:
            raise ValueError(
                f"Invalid Pole: {self.Pole}. "
                f"Must be one of {valid_poles}"
            )

        valid_line_types = ["GREAT_CIRCLE", "CART"]
        if self.LineType not in valid_line_types:
            raise ValueError(
                f"Invalid LineType: {self.LineType}. "
                f"Must be one of {valid_line_types}"
            )

        # Validate that destination grid coordinates are provided
        if self.DstGridLat is None or self.DstGridLon is None:
            if self.DstGridType is None and self.DstInputFileName is None:
                raise ValueError(
                    "Destination grid must be specified via DstGridLat/DstGridLon, "
                    "DstGridType, or DstInputFileName"
                )

    def __repr__(self):
        """String representation of options."""
        return f"RegridOptions(InterpMethod={self.InterpMethod}, Pole={self.Pole})"
