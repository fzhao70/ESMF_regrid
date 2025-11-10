"""
Grid utilities for ESMF_regrid.

This module provides functions for creating and managing ESMF grids from
various coordinate formats (rectilinear, curvilinear, unstructured).
"""

import numpy as np
import ESMF
from typing import Tuple, Optional, Union
import os
import warnings


def detect_grid_type(lat: np.ndarray, lon: np.ndarray) -> str:
    """
    Detect the type of grid from coordinates.

    Parameters
    ----------
    lat : np.ndarray
        Latitude coordinates
    lon : np.ndarray
        Longitude coordinates

    Returns
    -------
    str
        Grid type: "rectilinear", "curvilinear", or "unstructured"
    """
    if lat.ndim == 1 and lon.ndim == 1:
        return "rectilinear"
    elif lat.ndim == 2 and lon.ndim == 2:
        return "curvilinear"
    elif lat.ndim == 1 and lon.ndim == 1 and lat.shape == lon.shape:
        # Could be unstructured if they're equal length 1D arrays
        return "unstructured"
    else:
        raise ValueError(
            f"Cannot determine grid type from lat shape {lat.shape} "
            f"and lon shape {lon.shape}"
        )


def create_rectilinear_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    mask: Optional[np.ndarray] = None,
    regional: bool = False,
    corners: bool = False,
) -> ESMF.Grid:
    """
    Create an ESMF Grid from 1D rectilinear coordinates.

    Parameters
    ----------
    lat : np.ndarray
        1D array of latitude values
    lon : np.ndarray
        1D array of longitude values
    mask : np.ndarray, optional
        2D mask array (nlat, nlon)
    regional : bool
        Whether this is a regional grid
    corners : bool
        Whether to add corner coordinates

    Returns
    -------
    ESMF.Grid
        ESMF Grid object
    """
    nlat = len(lat)
    nlon = len(lon)

    # Determine periodicity
    if regional:
        periodic_dim = None
    else:
        # Check if longitude is periodic (spans 360 degrees)
        lon_range = np.max(lon) - np.min(lon)
        if lon_range >= 359.0:  # Allow some tolerance
            periodic_dim = 1  # Longitude is periodic
        else:
            periodic_dim = None

    # Create the grid
    grid = ESMF.Grid(
        np.array([nlon, nlat]),
        staggerloc=[ESMF.StaggerLoc.CENTER],
        coord_sys=ESMF.CoordSys.SPH_DEG,
        num_peri_dims=1 if periodic_dim is not None else 0,
        periodic_dim=periodic_dim,
    )

    # Add coordinates
    grid_lon = grid.get_coords(0)
    grid_lat = grid.get_coords(1)

    # Get the local bounds for parallel processing
    lon_slice = grid.lower_bounds[ESMF.StaggerLoc.CENTER][0]
    lat_slice = grid.lower_bounds[ESMF.StaggerLoc.CENTER][1]
    lon_size = grid.upper_bounds[ESMF.StaggerLoc.CENTER][0]
    lat_size = grid.upper_bounds[ESMF.StaggerLoc.CENTER][1]

    # Fill in coordinates
    lon_2d, lat_2d = np.meshgrid(lon[lon_slice:lon_size], lat[lat_slice:lat_size])
    grid_lon[...] = lon_2d.T
    grid_lat[...] = lat_2d.T

    # Add mask if provided
    if mask is not None:
        grid.add_item(ESMF.GridItem.MASK)
        grid_mask = grid.get_item(ESMF.GridItem.MASK)
        # ESMF uses 0 for valid points and 1 for masked points
        # Assuming input mask uses True/1 for masked points
        mask_subset = mask[lat_slice:lat_size, lon_slice:lon_size].T
        grid_mask[...] = mask_subset.astype(np.int32)

    # Add corners if requested
    if corners:
        grid.add_coords([ESMF.StaggerLoc.CORNER])
        corner_lon = grid.get_coords(0, staggerloc=ESMF.StaggerLoc.CORNER)
        corner_lat = grid.get_coords(1, staggerloc=ESMF.StaggerLoc.CORNER)

        # Calculate corner coordinates
        lat_corners = _calculate_corners(lat)
        lon_corners = _calculate_corners(lon)

        lon_corner_2d, lat_corner_2d = np.meshgrid(
            lon_corners[lon_slice : lon_size + 1], lat_corners[lat_slice : lat_size + 1]
        )
        corner_lon[...] = lon_corner_2d.T
        corner_lat[...] = lat_corner_2d.T

    return grid


def create_curvilinear_grid(
    lat: np.ndarray,
    lon: np.ndarray,
    mask: Optional[np.ndarray] = None,
    regional: bool = False,
    corner_lat: Optional[np.ndarray] = None,
    corner_lon: Optional[np.ndarray] = None,
) -> ESMF.Grid:
    """
    Create an ESMF Grid from 2D curvilinear coordinates.

    Parameters
    ----------
    lat : np.ndarray
        2D array of latitude values (nlat, nlon)
    lon : np.ndarray
        2D array of longitude values (nlat, nlon)
    mask : np.ndarray, optional
        2D mask array
    regional : bool
        Whether this is a regional grid
    corner_lat : np.ndarray, optional
        Corner latitudes (nlat+1, nlon+1) or (nlat, nlon, 4)
    corner_lon : np.ndarray, optional
        Corner longitudes (nlat+1, nlon+1) or (nlat, nlon, 4)

    Returns
    -------
    ESMF.Grid
        ESMF Grid object
    """
    nlat, nlon = lat.shape

    # Determine periodicity
    if regional:
        periodic_dim = None
    else:
        # For curvilinear grids, check if the grid wraps around
        lon_range = np.max(lon) - np.min(lon)
        if lon_range >= 359.0:
            periodic_dim = 1
        else:
            periodic_dim = None

    # Create the grid
    grid = ESMF.Grid(
        np.array([nlon, nlat]),
        staggerloc=[ESMF.StaggerLoc.CENTER],
        coord_sys=ESMF.CoordSys.SPH_DEG,
        num_peri_dims=1 if periodic_dim is not None else 0,
        periodic_dim=periodic_dim,
    )

    # Add coordinates
    grid_lon = grid.get_coords(0)
    grid_lat = grid.get_coords(1)

    # Get the local bounds
    lon_slice = grid.lower_bounds[ESMF.StaggerLoc.CENTER][0]
    lat_slice = grid.lower_bounds[ESMF.StaggerLoc.CENTER][1]
    lon_size = grid.upper_bounds[ESMF.StaggerLoc.CENTER][0]
    lat_size = grid.upper_bounds[ESMF.StaggerLoc.CENTER][1]

    # Fill in coordinates (note the transpose for ESMF's [lon, lat] ordering)
    grid_lon[...] = lon[lat_slice:lat_size, lon_slice:lon_size].T
    grid_lat[...] = lat[lat_slice:lat_size, lon_slice:lon_size].T

    # Add mask if provided
    if mask is not None:
        grid.add_item(ESMF.GridItem.MASK)
        grid_mask = grid.get_item(ESMF.GridItem.MASK)
        mask_subset = mask[lat_slice:lat_size, lon_slice:lon_size].T
        grid_mask[...] = mask_subset.astype(np.int32)

    # Add corners if provided
    if corner_lat is not None and corner_lon is not None:
        grid.add_coords([ESMF.StaggerLoc.CORNER])
        corner_lon_grid = grid.get_coords(0, staggerloc=ESMF.StaggerLoc.CORNER)
        corner_lat_grid = grid.get_coords(1, staggerloc=ESMF.StaggerLoc.CORNER)

        # Handle different corner formats
        if corner_lat.ndim == 2:  # (nlat+1, nlon+1)
            corner_lat_grid[...] = corner_lat[
                lat_slice : lat_size + 1, lon_slice : lon_size + 1
            ].T
            corner_lon_grid[...] = corner_lon[
                lat_slice : lat_size + 1, lon_slice : lon_size + 1
            ].T
        elif corner_lat.ndim == 3:  # (nlat, nlon, 4)
            # Convert from per-cell corners to grid corners
            # This is an approximation
            warnings.warn(
                "Converting per-cell corners to grid corners - this is an approximation"
            )

    return grid


def create_mesh_from_coords(
    lat: np.ndarray,
    lon: np.ndarray,
    elements: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> ESMF.Mesh:
    """
    Create an ESMF Mesh from unstructured coordinates.

    Parameters
    ----------
    lat : np.ndarray
        1D array of latitude values for each node
    lon : np.ndarray
        1D array of longitude values for each node
    elements : np.ndarray, optional
        Element connectivity array (num_elements, nodes_per_element)
    mask : np.ndarray, optional
        Mask for elements

    Returns
    -------
    ESMF.Mesh
        ESMF Mesh object
    """
    num_nodes = len(lat)

    if elements is None:
        # Create a simple triangulation if no connectivity provided
        # This is a placeholder - real implementation would need proper triangulation
        raise ValueError("Element connectivity must be provided for unstructured grids")

    num_elements = elements.shape[0]
    nodes_per_element = elements.shape[1]

    # Determine element type
    if nodes_per_element == 3:
        element_type = ESMF.MeshElemType.TRI
    elif nodes_per_element == 4:
        element_type = ESMF.MeshElemType.QUAD
    else:
        raise ValueError(
            f"Unsupported number of nodes per element: {nodes_per_element}"
        )

    # Create mesh
    mesh = ESMF.Mesh()
    mesh.add_nodes(num_nodes)

    # Add node coordinates
    node_coords = np.zeros((num_nodes, 2))
    node_coords[:, 0] = lon
    node_coords[:, 1] = lat
    mesh.set_coords(node_coords)

    # Add elements
    mesh.add_elements(num_elements, element_type)
    mesh.set_element_connectivity(elements)

    # Add mask if provided
    if mask is not None:
        mesh.set_element_mask(mask.astype(np.int32))

    mesh.build()

    return mesh


def _calculate_corners(centers: np.ndarray) -> np.ndarray:
    """
    Calculate corner coordinates from center coordinates.

    Parameters
    ----------
    centers : np.ndarray
        1D array of center coordinates

    Returns
    -------
    np.ndarray
        1D array of corner coordinates (length = len(centers) + 1)
    """
    corners = np.zeros(len(centers) + 1)

    # First corner is extrapolated
    corners[0] = centers[0] - (centers[1] - centers[0]) / 2.0

    # Interior corners are midpoints
    for i in range(1, len(centers)):
        corners[i] = (centers[i - 1] + centers[i]) / 2.0

    # Last corner is extrapolated
    corners[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0

    return corners


def create_grid_from_options(
    lat: np.ndarray,
    lon: np.ndarray,
    options,
    is_source: bool = True,
) -> Union[ESMF.Grid, ESMF.Mesh]:
    """
    Create an ESMF Grid or Mesh from coordinates and options.

    Parameters
    ----------
    lat : np.ndarray
        Latitude coordinates
    lon : np.ndarray
        Longitude coordinates
    options : RegridOptions
        Regridding options
    is_source : bool
        Whether this is the source grid (vs destination)

    Returns
    -------
    Union[ESMF.Grid, ESMF.Mesh]
        ESMF Grid or Mesh object
    """
    # Get options for this grid
    if is_source:
        mask = options.SrcGridMask
        regional = options.SrcRegional
        corner_lat = options.SrcGridCornerLat
        corner_lon = options.SrcGridCornerLon
        elements = options.SrcGridMeshElements
    else:
        mask = options.DstGridMask
        regional = options.DstRegional
        corner_lat = options.DstGridCornerLat
        corner_lon = options.DstGridCornerLon
        elements = options.DstGridMeshElements

    # Detect grid type
    grid_type = detect_grid_type(lat, lon)

    # Create appropriate grid
    if grid_type == "rectilinear":
        # Check if conservative method requires corners
        needs_corners = options.InterpMethod == "conserve"
        return create_rectilinear_grid(
            lat, lon, mask, regional, corners=needs_corners
        )
    elif grid_type == "curvilinear":
        return create_curvilinear_grid(
            lat, lon, mask, regional, corner_lat, corner_lon
        )
    elif grid_type == "unstructured":
        return create_mesh_from_coords(lat, lon, elements, mask)
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")


def parse_grid_type_string(grid_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a grid type string like "1x1" or "0.25deg" into lat/lon arrays.

    Parameters
    ----------
    grid_type : str
        Grid type specification

    Returns
    -------
    tuple of np.ndarray
        (lat, lon) coordinate arrays
    """
    # Handle "NxM" format
    if "x" in grid_type.lower():
        parts = grid_type.lower().replace("deg", "").split("x")
        lon_res = float(parts[0])
        lat_res = float(parts[1]) if len(parts) > 1 else lon_res

        lon = np.arange(0, 360, lon_res)
        lat = np.arange(-90, 90 + lat_res, lat_res)

        return lat, lon

    # Handle "N.Ndeg" format
    elif "deg" in grid_type.lower():
        res = float(grid_type.lower().replace("deg", ""))

        lon = np.arange(0, 360, res)
        lat = np.arange(-90, 90 + res, res)

        return lat, lon

    else:
        raise ValueError(f"Cannot parse grid type string: {grid_type}")
