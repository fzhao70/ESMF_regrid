"""
Tests for different grid types supported by ESMF_regrid.

This module tests rectilinear, curvilinear, regional, and global grids.
"""

import pytest
import numpy as np
import ESMF

from esmf_regrid import ESMF_regrid, RegridOptions
from esmf_regrid.grid_utils import (
    detect_grid_type,
    parse_grid_type_string,
    create_rectilinear_grid,
    create_curvilinear_grid,
)


@pytest.fixture(scope="module", autouse=True)
def setup_esmf():
    """Initialize ESMF for all tests."""
    if not ESMF.Manager().__class__._initialized:
        ESMF.Manager(debug=False)
    yield


class TestRectilinearGrids:
    """Tests for rectilinear grids."""

    def test_rectilinear_detection(self):
        """Test detection of rectilinear grids."""
        lat = np.linspace(-90, 90, 46)
        lon = np.linspace(0, 360, 91, endpoint=False)

        grid_type = detect_grid_type(lat, lon)
        assert grid_type == "rectilinear"

    def test_rectilinear_grid_creation(self):
        """Test creation of rectilinear ESMF grid."""
        lat = np.linspace(-90, 90, 46)
        lon = np.linspace(0, 360, 91, endpoint=False)

        grid = create_rectilinear_grid(lat, lon)

        assert grid is not None
        assert isinstance(grid, ESMF.Grid)

        # Cleanup
        grid.destroy()

    def test_global_rectilinear_regridding(self):
        """Test regridding between global rectilinear grids."""
        # Source: 4-degree grid
        src_lat = np.linspace(-90, 90, 46)
        src_lon = np.linspace(0, 360, 91, endpoint=False)

        # Destination: 2-degree grid
        dst_lat = np.linspace(-90, 90, 91)
        dst_lon = np.linspace(0, 360, 181, endpoint=False)

        # Create test data
        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcRegional = False
        opt.DstRegional = False

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_regional_rectilinear_regridding(self):
        """Test regridding between regional rectilinear grids."""
        # Source: Regional grid over North America
        src_lat = np.linspace(20, 60, 21)  # 20N to 60N
        src_lon = np.linspace(230, 300, 36)  # 130W to 60W

        # Destination: Finer regional grid
        dst_lat = np.linspace(20, 60, 41)
        dst_lon = np.linspace(230, 300, 71)

        # Create test data
        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d - 40) * 5) * np.cos(
            np.radians(lon_2d - 265) * 3
        )

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcRegional = True
        opt.DstRegional = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))


class TestCurvilinearGrids:
    """Tests for curvilinear grids."""

    def test_curvilinear_detection(self):
        """Test detection of curvilinear grids."""
        nlat, nlon = 30, 40
        lat = np.random.randn(nlat, nlon) * 20 + 40
        lon = np.random.randn(nlat, nlon) * 30 + 180

        grid_type = detect_grid_type(lat, lon)
        assert grid_type == "curvilinear"

    def test_curvilinear_grid_creation(self):
        """Test creation of curvilinear ESMF grid."""
        nlat, nlon = 20, 30

        # Create a simple curvilinear grid (slightly distorted regular grid)
        lat_1d = np.linspace(-90, 90, nlat)
        lon_1d = np.linspace(0, 360, nlon, endpoint=False)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        # Add some curvature
        lat_2d = lat_2d + 2 * np.sin(np.radians(lon_2d) * 2)

        grid = create_curvilinear_grid(lat_2d, lon_2d)

        assert grid is not None
        assert isinstance(grid, ESMF.Grid)

        # Cleanup
        grid.destroy()

    def test_curvilinear_to_rectilinear_regridding(self):
        """Test regridding from curvilinear to rectilinear grid."""
        # Source: Curvilinear grid
        nlat, nlon = 30, 40
        lat_1d = np.linspace(-60, 60, nlat)
        lon_1d = np.linspace(0, 360, nlon, endpoint=False)
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)

        # Add curvature
        src_lat = lat_2d + 2 * np.sin(np.radians(lon_2d) * 2)
        src_lon = lon_2d + 3 * np.cos(np.radians(lat_2d) * 1.5)

        # Create test data
        data = np.sin(np.radians(src_lat) * 3) + np.cos(np.radians(src_lon) * 2)

        # Destination: Regular rectilinear grid
        dst_lat = np.linspace(-60, 60, 61)
        dst_lon = np.linspace(0, 360, 72, endpoint=False)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcRegional = False
        opt.DstRegional = False

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_rectilinear_to_curvilinear_regridding(self):
        """Test regridding from rectilinear to curvilinear grid."""
        # Source: Rectilinear grid
        src_lat = np.linspace(-60, 60, 31)
        src_lon = np.linspace(0, 360, 61, endpoint=False)

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

        # Destination: Curvilinear grid
        nlat, nlon = 25, 35
        dst_lat_1d = np.linspace(-60, 60, nlat)
        dst_lon_1d = np.linspace(0, 360, nlon, endpoint=False)
        dst_lon_2d, dst_lat_2d = np.meshgrid(dst_lon_1d, dst_lat_1d)

        # Add curvature
        dst_lat = dst_lat_2d + 2 * np.sin(np.radians(dst_lon_2d) * 2)
        dst_lon = dst_lon_2d + 3 * np.cos(np.radians(dst_lat_2d) * 1.5)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data, opt)

        assert result.shape == dst_lat.shape
        assert not np.all(np.isnan(result))


class TestGridTypeStrings:
    """Tests for grid type string parsing."""

    def test_parse_degree_format(self):
        """Test parsing of degree format grid types."""
        lat, lon = parse_grid_type_string("1deg")

        assert len(lat) == 181  # -90 to 90 in 1-degree steps
        assert len(lon) == 360  # 0 to 359 in 1-degree steps
        assert np.allclose(lat[0], -90)
        assert np.allclose(lat[-1], 90)
        assert np.allclose(lon[0], 0)

    def test_parse_nxm_format(self):
        """Test parsing of NxM format grid types."""
        lat, lon = parse_grid_type_string("2x2")

        assert len(lon) == 180  # 0 to 358 in 2-degree steps
        assert len(lat) == 91  # -90 to 90 in 2-degree steps

    def test_parse_fractional_degree(self):
        """Test parsing of fractional degree format."""
        lat, lon = parse_grid_type_string("0.25deg")

        assert len(lon) == 1440  # 0 to 359.75 in 0.25-degree steps
        assert len(lat) == 721  # -90 to 90 in 0.25-degree steps

    def test_regrid_with_grid_type_string(self):
        """Test regridding using grid type strings."""
        # Source data on 2-degree grid
        src_lat = np.linspace(-90, 90, 91)
        src_lon = np.linspace(0, 360, 181, endpoint=False)

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridType = "1deg"  # Use string specification

        result = ESMF_regrid(data, opt)

        # Check that output matches 1-degree grid
        assert result.shape[0] == 181  # latitude
        assert result.shape[1] == 360  # longitude
        assert not np.all(np.isnan(result))


class TestDifferentResolutions:
    """Tests for regridding between different resolutions."""

    def test_coarse_to_fine(self):
        """Test regridding from coarse to fine resolution."""
        # Coarse source
        src_lat = np.linspace(-90, 90, 23)  # 8-degree
        src_lon = np.linspace(0, 360, 46, endpoint=False)

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 2)

        # Fine destination
        dst_lat = np.linspace(-90, 90, 181)  # 1-degree
        dst_lon = np.linspace(0, 360, 360, endpoint=False)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_fine_to_coarse(self):
        """Test regridding from fine to coarse resolution."""
        # Fine source
        src_lat = np.linspace(-90, 90, 181)  # 1-degree
        src_lon = np.linspace(0, 360, 360, endpoint=False)

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 5) + np.cos(np.radians(lon_2d) * 3)

        # Coarse destination
        dst_lat = np.linspace(-90, 90, 46)  # 4-degree
        dst_lon = np.linspace(0, 360, 91, endpoint=False)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_same_resolution(self):
        """Test regridding between grids with same resolution but different origins."""
        # Source grid: 0 to 360
        src_lat = np.linspace(-90, 90, 91)
        src_lon = np.linspace(0, 360, 181, endpoint=False)

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Destination grid: -180 to 180
        dst_lat = np.linspace(-90, 90, 91)
        dst_lon = np.linspace(-180, 180, 181, endpoint=False)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))


class TestSpecialGridConfigurations:
    """Tests for special grid configurations."""

    def test_pole_to_pole_grid(self):
        """Test grid that includes both poles."""
        src_lat = np.linspace(-90, 90, 91)
        src_lon = np.linspace(0, 360, 181, endpoint=False)

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.cos(np.radians(lat_2d) * 2)

        dst_lat = np.linspace(-90, 90, 181)
        dst_lon = np.linspace(0, 360, 361, endpoint=False)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))

    def test_tropical_band(self):
        """Test regridding in tropical band (no poles)."""
        src_lat = np.linspace(-30, 30, 31)
        src_lon = np.linspace(0, 360, 181, endpoint=False)

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 6)

        dst_lat = np.linspace(-30, 30, 61)
        dst_lon = np.linspace(0, 360, 361, endpoint=False)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcRegional = True
        opt.DstRegional = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
