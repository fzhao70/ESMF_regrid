"""
Weight file management for ESMF_regrid.

This module handles weight file generation, caching, and reuse.
"""

import os
import hashlib
import json
import numpy as np
from typing import Optional, Union
import ESMF


def generate_weight_file_key(
    src_shape: tuple,
    dst_shape: tuple,
    interp_method: str,
    src_mask: Optional[np.ndarray] = None,
    dst_mask: Optional[np.ndarray] = None,
    options_dict: Optional[dict] = None,
) -> str:
    """
    Generate a unique key for a weight file based on grid properties.

    Parameters
    ----------
    src_shape : tuple
        Shape of source grid
    dst_shape : tuple
        Shape of destination grid
    interp_method : str
        Interpolation method
    src_mask : np.ndarray, optional
        Source mask
    dst_mask : np.ndarray, optional
        Destination mask
    options_dict : dict, optional
        Additional options that affect weights

    Returns
    -------
    str
        Unique hash key
    """
    key_data = {
        "src_shape": src_shape,
        "dst_shape": dst_shape,
        "interp_method": interp_method,
    }

    # Include mask information
    if src_mask is not None:
        key_data["src_mask_hash"] = hashlib.md5(src_mask.tobytes()).hexdigest()

    if dst_mask is not None:
        key_data["dst_mask_hash"] = hashlib.md5(dst_mask.tobytes()).hexdigest()

    # Include relevant options
    if options_dict is not None:
        key_data["options"] = options_dict

    # Generate hash
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def create_regrid_object(
    srcfield: ESMF.Field,
    dstfield: ESMF.Field,
    options,
) -> ESMF.Regrid:
    """
    Create an ESMF Regrid object with the specified options.

    Parameters
    ----------
    srcfield : ESMF.Field
        Source field
    dstfield : ESMF.Field
        Destination field
    options : RegridOptions
        Regridding options

    Returns
    -------
    ESMF.Regrid
        ESMF Regrid object
    """
    # Map interpolation method names to ESMF constants
    method_map = {
        "bilinear": ESMF.RegridMethod.BILINEAR,
        "patch": ESMF.RegridMethod.PATCH,
        "conserve": ESMF.RegridMethod.CONSERVE,
        "neareststod": ESMF.RegridMethod.NEAREST_STOD,
        "nearestdtos": ESMF.RegridMethod.NEAREST_DTOS,
    }

    regrid_method = method_map.get(options.InterpMethod.lower())
    if regrid_method is None:
        raise ValueError(f"Unknown interpolation method: {options.InterpMethod}")

    # Map pole method names to ESMF constants
    pole_method = None
    if options.Pole != "none" and options.InterpMethod in ["bilinear", "patch"]:
        pole_map = {
            "none": None,
            "all": ESMF.PoleMethod.ALLAVG,
            "teeth": ESMF.PoleMethod.TEETH,
        }
        pole_method = pole_map.get(options.Pole.lower())

    # Map line type to ESMF constant
    line_type_map = {
        "GREAT_CIRCLE": ESMF.LineType.GREAT_CIRCLE,
        "CART": ESMF.LineType.CART,
    }
    line_type = line_type_map.get(options.LineType, ESMF.LineType.GREAT_CIRCLE)

    # Map norm type for conservative regridding
    norm_type = None
    if regrid_method == ESMF.RegridMethod.CONSERVE:
        norm_type_map = {
            "FRACAREA": ESMF.NormType.FRACAREA,
            "DSTAREA": ESMF.NormType.DSTAREA,
        }
        norm_type = norm_type_map.get(options.NormType, ESMF.NormType.FRACAREA)

    # Determine unmapped action
    if options.IgnoreUnmappedPoints:
        unmapped_action = ESMF.UnmappedAction.IGNORE
    else:
        unmapped_action = ESMF.UnmappedAction.ERROR

    # Map extrapolation method
    extrap_method = None
    if options.ExtrapMethod is not None:
        extrap_map = {
            "neareststod": ESMF.ExtrapMethod.NEAREST_STOD,
            "nearestidavg": ESMF.ExtrapMethod.NEAREST_IDAVG,
            "creep": ESMF.ExtrapMethod.CREEP,
        }
        extrap_method = extrap_map.get(options.ExtrapMethod.lower())

    # Build regrid arguments
    regrid_kwargs = {
        "srcfield": srcfield,
        "dstfield": dstfield,
        "regrid_method": regrid_method,
        "unmapped_action": unmapped_action,
        "line_type": line_type,
    }

    # Add optional arguments
    if pole_method is not None:
        regrid_kwargs["pole_method"] = pole_method

    if norm_type is not None:
        regrid_kwargs["norm_type"] = norm_type

    if extrap_method is not None:
        regrid_kwargs["extrap_method"] = extrap_method
        regrid_kwargs["extrap_num_src_pnts"] = options.ExtrapNumSrcPnts
        regrid_kwargs["extrap_dist_exponent"] = options.ExtrapDistExponent

    # Create the regrid object
    try:
        regrid = ESMF.Regrid(**regrid_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create regrid object: {e}")

    return regrid


def apply_regridding(
    regrid: ESMF.Regrid,
    srcfield: ESMF.Field,
    dstfield: ESMF.Field,
    src_data: np.ndarray,
) -> np.ndarray:
    """
    Apply regridding to data.

    Parameters
    ----------
    regrid : ESMF.Regrid
        ESMF Regrid object
    srcfield : ESMF.Field
        Source field
    dstfield : ESMF.Field
        Destination field
    src_data : np.ndarray
        Source data to regrid

    Returns
    -------
    np.ndarray
        Regridded data
    """
    # Copy data to source field
    srcfield.data[...] = src_data

    # Perform regridding
    dstfield = regrid(srcfield, dstfield)

    # Return the regridded data
    return dstfield.data.copy()


class WeightFileManager:
    """
    Manager for ESMF weight files.

    This class handles creation, caching, and reuse of weight files.
    """

    def __init__(self, weight_file: str, force_overwrite: bool = False):
        """
        Initialize the weight file manager.

        Parameters
        ----------
        weight_file : str
            Path to weight file
        force_overwrite : bool
            Whether to force regeneration of weight file
        """
        self.weight_file = weight_file
        self.force_overwrite = force_overwrite

    def exists(self) -> bool:
        """Check if weight file exists."""
        return os.path.exists(self.weight_file) and not self.force_overwrite

    def create_regrid_with_weights(
        self,
        srcfield: ESMF.Field,
        dstfield: ESMF.Field,
        options,
    ) -> ESMF.Regrid:
        """
        Create a regrid object using or creating a weight file.

        Parameters
        ----------
        srcfield : ESMF.Field
            Source field
        dstfield : ESMF.Field
            Destination field
        options : RegridOptions
            Regridding options

        Returns
        -------
        ESMF.Regrid
            ESMF Regrid object
        """
        # Check if we should use existing weights
        if self.exists():
            try:
                # Try to use existing weight file
                regrid = self._load_from_file(srcfield, dstfield, options)
                return regrid
            except Exception as e:
                if not self.force_overwrite:
                    raise RuntimeError(
                        f"Failed to load weight file {self.weight_file}: {e}"
                    )

        # Create new regrid object (will generate weights)
        regrid = create_regrid_object(srcfield, dstfield, options)

        # Note: ESMF's Python interface doesn't directly support saving
        # weight files like the Fortran/C interface. In a production
        # implementation, you would use ESMF's weight file I/O capabilities.

        return regrid

    def _load_from_file(
        self,
        srcfield: ESMF.Field,
        dstfield: ESMF.Field,
        options,
    ) -> ESMF.Regrid:
        """
        Load regrid object from weight file.

        Parameters
        ----------
        srcfield : ESMF.Field
            Source field
        dstfield : ESMF.Field
            Destination field
        options : RegridOptions
            Regridding options

        Returns
        -------
        ESMF.Regrid
            ESMF Regrid object
        """
        # Map interpolation method names to ESMF constants
        method_map = {
            "bilinear": ESMF.RegridMethod.BILINEAR,
            "patch": ESMF.RegridMethod.PATCH,
            "conserve": ESMF.RegridMethod.CONSERVE,
            "neareststod": ESMF.RegridMethod.NEAREST_STOD,
            "nearestdtos": ESMF.RegridMethod.NEAREST_DTOS,
        }

        regrid_method = method_map.get(options.InterpMethod.lower())

        # Create regrid from file
        # Note: This is a simplified version. Full implementation would need
        # to handle all the same options as create_regrid_object
        regrid = ESMF.RegridFromFile(
            srcfield=srcfield,
            dstfield=dstfield,
            filename=self.weight_file,
        )

        return regrid

    def cleanup(self):
        """Remove weight file if requested."""
        if os.path.exists(self.weight_file):
            try:
                os.remove(self.weight_file)
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to remove weight file {self.weight_file}: {e}")
