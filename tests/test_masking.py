"""
Tests for masking and missing value handling in ESMF_regrid.

This module tests source/destination masks, missing values, and their combinations.
"""

import pytest
import numpy as np
import ESMF

from esmf_regrid import ESMF_regrid, RegridOptions


@pytest.fixture(scope="module", autouse=True)
def setup_esmf():
    """Initialize ESMF for all tests."""
    if not ESMF.Manager().__class__._initialized:
        ESMF.Manager(debug=False)
    yield


@pytest.fixture
def source_grid_coords():
    """Create source grid coordinates."""
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    return src_lat, src_lon


@pytest.fixture
def dest_grid_coords():
    """Create destination grid coordinates."""
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)
    return dst_lat, dst_lon


class TestSourceMasking:
    """Tests for source grid masking."""

    def test_source_mask_basic(self, source_grid_coords, dest_grid_coords):
        """Test basic source masking."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Create data
        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Create mask (mask ocean, keep land)
        src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
        src_mask[:, :45] = 1  # Mask western half

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        # Some points should be affected by the mask
        assert np.any(np.isnan(result) | (result == 0))

    def test_source_mask_all_masked(self, source_grid_coords, dest_grid_coords):
        """Test with all source points masked."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Mask everything
        src_mask = np.ones((len(src_lat), len(src_lon)), dtype=np.int32)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        # Should be all NaN or zero since everything is masked
        assert result.shape == (len(dst_lat), len(dst_lon))

    def test_source_mask_circular_region(self, source_grid_coords, dest_grid_coords):
        """Test masking a circular region."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

        # Create circular mask around center
        center_lat, center_lon = 0, 180
        radius = 30  # degrees

        # Calculate distance from center
        dlat = lat_2d - center_lat
        dlon = lon_2d - center_lon
        # Handle longitude wraparound
        dlon = np.where(dlon > 180, dlon - 360, dlon)
        dlon = np.where(dlon < -180, dlon + 360, dlon)

        distance = np.sqrt(dlat**2 + dlon**2)
        src_mask = (distance < radius).astype(np.int32)  # Mask inside circle

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))


class TestDestinationMasking:
    """Tests for destination grid masking."""

    def test_destination_mask_basic(self, source_grid_coords, dest_grid_coords):
        """Test basic destination masking."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Create destination mask
        dst_mask = np.zeros((len(dst_lat), len(dst_lon)), dtype=np.int32)
        dst_mask[:45, :] = 1  # Mask southern hemisphere

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.DstGridMask = dst_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        # Masked region should be affected
        assert np.any(np.isnan(result[:45, :]) | (result[:45, :] == 0))


class TestMissingValues:
    """Tests for missing value handling."""

    def test_missing_values_in_source(self, source_grid_coords, dest_grid_coords):
        """Test handling of missing values in source data."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Add missing values
        data[10:15, 20:30] = np.nan

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))

    def test_missing_value_specification(self, source_grid_coords, dest_grid_coords):
        """Test explicit missing value specification."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Use specific missing value
        missing_val = -999.0
        data[10:15, 20:30] = missing_val

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcMissingValue = missing_val
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))

    def test_missing_values_near_poles(self, dest_grid_coords):
        """Test handling of missing values near poles."""
        # Create source grid with missing values at poles
        src_lat = np.linspace(-90, 90, 46)
        src_lon = np.linspace(0, 360, 91, endpoint=False)
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Mask poles
        data[0, :] = np.nan  # South pole
        data[-1, :] = np.nan  # North pole

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))


class TestCombinedMaskingAndMissingValues:
    """Tests for combined masking and missing value handling."""

    def test_mask_and_missing_values(self, source_grid_coords, dest_grid_coords):
        """Test with both mask and missing values."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Add missing values
        data[5:10, 10:20] = np.nan

        # Add mask
        src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
        src_mask[20:25, 30:40] = 1

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))

    def test_source_and_dest_masks(self, source_grid_coords, dest_grid_coords):
        """Test with both source and destination masks."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Source mask
        src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
        src_mask[:, :30] = 1

        # Destination mask
        dst_mask = np.zeros((len(dst_lat), len(dst_lon)), dtype=np.int32)
        dst_mask[:, -60:] = 1

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.DstGridMask = dst_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))


class TestMaskingWithDifferentMethods:
    """Tests for masking with different interpolation methods."""

    def test_mask_with_conservative(self, source_grid_coords, dest_grid_coords):
        """Test masking with conservative interpolation."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.ones_like(lat_2d)

        # Create land-sea mask
        src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
        src_mask[:, :45] = 1  # Mask half

        opt = RegridOptions()
        opt.InterpMethod = "conserve"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))

    def test_mask_with_nearest(self, source_grid_coords, dest_grid_coords):
        """Test masking with nearest neighbor interpolation."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Create mask
        src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
        src_mask[15:30, 30:60] = 1

        opt = RegridOptions()
        opt.InterpMethod = "neareststod"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))

    def test_mask_with_patch(self, source_grid_coords, dest_grid_coords):
        """Test masking with patch interpolation."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.sin(np.radians(lat_2d) * 3)

        # Create mask
        src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
        src_mask[10:20, 20:40] = 1

        opt = RegridOptions()
        opt.InterpMethod = "patch"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))


class TestMaskingMultiDimensionalData:
    """Tests for masking with multi-dimensional data."""

    def test_mask_3d_data(self, source_grid_coords, dest_grid_coords):
        """Test masking with 3D data (time, lat, lon)."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Create 3D data
        n_time = 5
        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data_3d = np.stack([np.sin(np.radians(lat_2d) * 3) * (i + 1) for i in range(n_time)])

        # Create mask
        src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
        src_mask[:, :30] = 1

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data_3d, opt)

        assert result.shape == (n_time, len(dst_lat), len(dst_lon))

    def test_mask_4d_data(self, source_grid_coords, dest_grid_coords):
        """Test masking with 4D data (time, level, lat, lon)."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Create 4D data
        n_time = 3
        n_level = 4
        lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
        data_4d = np.stack(
            [[np.sin(np.radians(lat_2d) * 3) * (i + 1) * (j + 1)
              for j in range(n_level)] for i in range(n_time)]
        )

        # Create mask
        src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
        src_mask[15:30, :] = 1

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.SrcGridMask = src_mask
        opt.IgnoreUnmappedPoints = True

        result = ESMF_regrid(data_4d, opt)

        assert result.shape == (n_time, n_level, len(dst_lat), len(dst_lon))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
