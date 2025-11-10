"""
Core ESMF_regrid functionality.

This module implements the main ESMF_regrid function, compatible with the
NCL ESMF_regrid interface.
"""

import numpy as np
import ESMF
from typing import Union, Optional, Dict, Any
import warnings
import os

from .options import RegridOptions
from .grid_utils import (
    create_grid_from_options,
    detect_grid_type,
    parse_grid_type_string,
)
from .weights import WeightFileManager, apply_regridding


def detect_coordinates(data: np.ndarray, options: RegridOptions) -> tuple:
    """
    Detect or extract coordinate information from data.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    options : RegridOptions
        Regridding options

    Returns
    -------
    tuple
        (src_lat, src_lon, dst_lat, dst_lon)
    """
    # Source coordinates
    if options.SrcGridLat is not None and options.SrcGridLon is not None:
        src_lat = options.SrcGridLat
        src_lon = options.SrcGridLon
    elif options.SrcGridType is not None:
        src_lat, src_lon = parse_grid_type_string(options.SrcGridType)
    else:
        # Try to infer from data shape
        # Assume last two dimensions are lat, lon
        if data.ndim >= 2:
            nlat, nlon = data.shape[-2:]
            # Create default coordinates
            src_lat = np.linspace(-90, 90, nlat)
            src_lon = np.linspace(0, 360, nlon, endpoint=False)
            warnings.warn(
                f"Source coordinates not specified. Using default: "
                f"lat={nlat}, lon={nlon}"
            )
        else:
            raise ValueError(
                "Cannot infer source coordinates. Please specify SrcGridLat/SrcGridLon"
            )

    # Destination coordinates
    if options.DstGridLat is not None and options.DstGridLon is not None:
        dst_lat = options.DstGridLat
        dst_lon = options.DstGridLon
    elif options.DstGridType is not None:
        dst_lat, dst_lon = parse_grid_type_string(options.DstGridType)
    else:
        raise ValueError(
            "Destination coordinates must be specified via DstGridLat/DstGridLon or DstGridType"
        )

    return src_lat, src_lon, dst_lat, dst_lon


def prepare_data_for_regridding(
    data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray
) -> tuple:
    """
    Prepare data for regridding by reshaping if necessary.

    Parameters
    ----------
    data : np.ndarray
        Input data
    src_lat : np.ndarray
        Source latitude coordinates
    src_lon : np.ndarray
        Source longitude coordinates

    Returns
    -------
    tuple
        (reshaped_data, original_shape, extra_dims)
    """
    original_shape = data.shape

    # Determine expected spatial dimensions
    if src_lat.ndim == 1 and src_lon.ndim == 1:
        # Rectilinear grid
        spatial_shape = (len(src_lat), len(src_lon))
    elif src_lat.ndim == 2 and src_lon.ndim == 2:
        # Curvilinear grid
        spatial_shape = src_lat.shape
    else:
        # Unstructured
        spatial_shape = (len(src_lat),)

    # Check if data has extra dimensions
    if len(spatial_shape) == 2:
        expected_spatial_size = spatial_shape[0] * spatial_shape[1]
        if data.ndim == 2:
            # Data is 2D (nlat, nlon)
            extra_dims = ()
            if data.shape != spatial_shape:
                raise ValueError(
                    f"Data shape {data.shape} does not match grid shape {spatial_shape}"
                )
        elif data.ndim > 2:
            # Data has extra dimensions (e.g., time, level, lat, lon)
            # Verify that last two dimensions match spatial shape
            if data.shape[-2:] != spatial_shape:
                raise ValueError(
                    f"Data spatial dimensions {data.shape[-2:]} do not match "
                    f"grid shape {spatial_shape}"
                )
            extra_dims = data.shape[:-2]
            # Reshape to (extra_dims_flattened, nlat, nlon)
            data = data.reshape(-1, *spatial_shape)
        else:
            raise ValueError(f"Data has insufficient dimensions: {data.shape}")
    else:
        # Unstructured grid
        if data.shape[-1] != spatial_shape[0]:
            raise ValueError(
                f"Data size {data.shape[-1]} does not match grid size {spatial_shape[0]}"
            )
        if data.ndim > 1:
            extra_dims = data.shape[:-1]
            data = data.reshape(-1, spatial_shape[0])
        else:
            extra_dims = ()

    return data, original_shape, extra_dims


def create_fields(
    src_grid: Union[ESMF.Grid, ESMF.Mesh],
    dst_grid: Union[ESMF.Grid, ESMF.Mesh],
) -> tuple:
    """
    Create ESMF fields for source and destination grids.

    Parameters
    ----------
    src_grid : Union[ESMF.Grid, ESMF.Mesh]
        Source grid
    dst_grid : Union[ESMF.Grid, ESMF.Mesh]
        Destination grid

    Returns
    -------
    tuple
        (srcfield, dstfield)
    """
    srcfield = ESMF.Field(src_grid, name="source_field")
    dstfield = ESMF.Field(dst_grid, name="destination_field")

    return srcfield, dstfield


def handle_missing_values(
    data: np.ndarray, options: RegridOptions, is_source: bool = True
) -> np.ndarray:
    """
    Handle missing values in data by creating/updating mask.

    Parameters
    ----------
    data : np.ndarray
        Input data
    options : RegridOptions
        Regridding options
    is_source : bool
        Whether this is source data

    Returns
    -------
    np.ndarray
        Mask array (True for masked points)
    """
    missing_value = (
        options.SrcMissingValue if is_source else options.DstMissingValue
    )

    if missing_value is not None:
        # Create mask from missing values
        mask = np.isclose(data, missing_value) | np.isnan(data)
    else:
        # Just check for NaN
        mask = np.isnan(data)

    # Combine with existing mask if present
    existing_mask = options.SrcGridMask if is_source else options.DstGridMask
    if existing_mask is not None:
        mask = mask | existing_mask.astype(bool)

    return mask.astype(np.int32)


def reshape_output(
    regridded_data: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    extra_dims: tuple,
    original_shape: tuple,
) -> np.ndarray:
    """
    Reshape regridded data back to original dimensionality.

    Parameters
    ----------
    regridded_data : np.ndarray
        Regridded data
    dst_lat : np.ndarray
        Destination latitude
    dst_lon : np.ndarray
        Destination longitude
    extra_dims : tuple
        Extra dimensions from input
    original_shape : tuple
        Original data shape

    Returns
    -------
    np.ndarray
        Reshaped regridded data
    """
    # Determine output spatial shape
    if dst_lat.ndim == 1 and dst_lon.ndim == 1:
        spatial_shape = (len(dst_lat), len(dst_lon))
    elif dst_lat.ndim == 2 and dst_lon.ndim == 2:
        spatial_shape = dst_lat.shape
    else:
        spatial_shape = (len(dst_lat),)

    # Reshape back to include extra dimensions
    if extra_dims:
        output_shape = extra_dims + spatial_shape
        regridded_data = regridded_data.reshape(output_shape)

    return regridded_data


def ESMF_regrid(
    data: np.ndarray,
    opt: Optional[Union[RegridOptions, dict, bool]] = None,
) -> np.ndarray:
    """
    Regrid data from source grid to destination grid using ESMF.

    This function provides an interface similar to NCL's ESMF_regrid function,
    supporting regridding between rectilinear, curvilinear, and unstructured grids.

    Parameters
    ----------
    data : np.ndarray
        Input data to be regridded. Can be any dimensionality; for curvilinear
        or rectilinear grids, the rightmost two dimensions must be nlat × nlon.
    opt : RegridOptions, dict, bool, or None
        Regridding options. Can be:
        - RegridOptions object
        - Dictionary of options
        - False to disable all options (not recommended)
        - None to use defaults

    Returns
    -------
    np.ndarray
        Regridded data with same number of dimensions as input.
        Shape will be (..., dst_nlat, dst_nlon) where ... represents
        any leading dimensions from the input.

    Raises
    ------
    ValueError
        If options are invalid or required coordinates are missing
    RuntimeError
        If regridding fails

    Examples
    --------
    >>> # Simple regridding with default options
    >>> opt = RegridOptions()
    >>> opt.InterpMethod = "bilinear"
    >>> opt.DstGridLat = np.linspace(-90, 90, 180)
    >>> opt.DstGridLon = np.linspace(0, 360, 360, endpoint=False)
    >>> data_out = ESMF_regrid(data_in, opt)

    >>> # Using grid type specification
    >>> opt = RegridOptions()
    >>> opt.DstGridType = "1x1"
    >>> data_out = ESMF_regrid(data_in, opt)

    >>> # Conservative regridding
    >>> opt = RegridOptions()
    >>> opt.InterpMethod = "conserve"
    >>> opt.DstGridLat = dst_lat
    >>> opt.DstGridLon = dst_lon
    >>> data_out = ESMF_regrid(data_in, opt)

    Notes
    -----
    - Initialize ESMF before calling this function
    - Supports bilinear, patch, conserve, neareststod, and nearestdtos methods
    - Weight files can be reused for multiple regridding operations
    - For conservative regridding, grid corners are required
    - Missing values in data should be set to NaN or specified via SrcMissingValue
    """
    # Initialize ESMF if not already done
    if not ESMF.Manager().__class__._initialized:
        ESMF.Manager(debug=False)

    # Parse options
    if opt is False or opt is None:
        options = RegridOptions()
    elif isinstance(opt, dict):
        options = RegridOptions.from_dict(opt)
    elif isinstance(opt, RegridOptions):
        options = opt
    else:
        raise ValueError(f"Invalid options type: {type(opt)}")

    # Validate options
    options.validate()

    # Detect or get coordinates
    src_lat, src_lon, dst_lat, dst_lon = detect_coordinates(data, options)

    # Update options with detected coordinates if not already set
    if options.SrcGridLat is None:
        options.SrcGridLat = src_lat
    if options.SrcGridLon is None:
        options.SrcGridLon = src_lon

    # Handle missing values and create/update masks
    if data.ndim >= 2:
        data_2d = data.reshape(-1, *data.shape[-2:])
        src_mask = handle_missing_values(data_2d[0], options, is_source=True)
        if src_mask.any():
            if options.SrcGridMask is None:
                options.SrcGridMask = src_mask
            else:
                options.SrcGridMask = options.SrcGridMask | src_mask

    # Prepare data for regridding
    data_reshaped, original_shape, extra_dims = prepare_data_for_regridding(
        data, src_lat, src_lon
    )

    # Create grids
    try:
        src_grid = create_grid_from_options(src_lat, src_lon, options, is_source=True)
        dst_grid = create_grid_from_options(dst_lat, dst_lon, options, is_source=False)
    except Exception as e:
        raise RuntimeError(f"Failed to create grids: {e}")

    # Create fields
    srcfield, dstfield = create_fields(src_grid, dst_grid)

    # Create weight file manager
    weight_manager = WeightFileManager(
        options.WgtFileName, options.ForceOverwrite
    )

    try:
        # Create regrid object
        regrid = weight_manager.create_regrid_with_weights(srcfield, dstfield, options)

        # Perform regridding for each slice of extra dimensions
        if data_reshaped.ndim == 2 or (data_reshaped.ndim == 3 and data_reshaped.shape[0] == 1):
            # Single slice
            if data_reshaped.ndim == 3:
                data_slice = data_reshaped[0]
            else:
                data_slice = data_reshaped

            regridded = apply_regridding(regrid, srcfield, dstfield, data_slice)

            # Reshape to destination grid shape
            if dst_lat.ndim == 1 and dst_lon.ndim == 1:
                regridded = regridded.T  # Transpose back from ESMF [lon, lat] to [lat, lon]

            regridded_data = regridded

        else:
            # Multiple slices
            n_slices = data_reshaped.shape[0]

            # Determine output shape for one slice
            if dst_lat.ndim == 1 and dst_lon.ndim == 1:
                slice_shape = (len(dst_lat), len(dst_lon))
            elif dst_lat.ndim == 2:
                slice_shape = dst_lat.shape
            else:
                slice_shape = (len(dst_lat),)

            # Allocate output array
            output_data = np.zeros((n_slices,) + slice_shape, dtype=data.dtype)

            # Regrid each slice
            for i in range(n_slices):
                data_slice = data_reshaped[i]
                regridded = apply_regridding(regrid, srcfield, dstfield, data_slice)

                if dst_lat.ndim == 1 and dst_lon.ndim == 1:
                    regridded = regridded.T

                output_data[i] = regridded

            regridded_data = output_data

        # Destroy regrid object to free memory
        regrid.destroy()

    except Exception as e:
        raise RuntimeError(f"Regridding failed: {e}")

    finally:
        # Cleanup
        try:
            srcfield.destroy()
            dstfield.destroy()
            if isinstance(src_grid, ESMF.Grid):
                src_grid.destroy()
            if isinstance(dst_grid, ESMF.Grid):
                dst_grid.destroy()
        except:
            pass

        # Remove weight file if requested
        if options.RemoveWgtFile:
            weight_manager.cleanup()

    # Reshape output back to original dimensionality
    regridded_data = reshape_output(
        regridded_data, dst_lat, dst_lon, extra_dims, original_shape
    )

    # Handle return type
    if options.ReturnDouble:
        regridded_data = regridded_data.astype(np.float64)
    elif data.dtype == np.float32:
        regridded_data = regridded_data.astype(np.float32)

    return regridded_data


def ESMF_regrid_with_coords(
    data: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    method: str = "bilinear",
    **kwargs,
) -> np.ndarray:
    """
    Convenience function for regridding with explicit coordinates.

    Parameters
    ----------
    data : np.ndarray
        Input data
    src_lat : np.ndarray
        Source latitude coordinates
    src_lon : np.ndarray
        Source longitude coordinates
    dst_lat : np.ndarray
        Destination latitude coordinates
    dst_lon : np.ndarray
        Destination longitude coordinates
    method : str
        Interpolation method (default: "bilinear")
    **kwargs
        Additional options passed to RegridOptions

    Returns
    -------
    np.ndarray
        Regridded data
    """
    opt = RegridOptions()
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.InterpMethod = method

    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(opt, key):
            setattr(opt, key, value)
        else:
            warnings.warn(f"Unknown option: {key}")

    return ESMF_regrid(data, opt)
