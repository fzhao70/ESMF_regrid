"""
Tests for all interpolation methods supported by ESMF_regrid.

This module tests bilinear, patch, conserve, neareststod, and nearestdtos
interpolation methods.
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
    # ESMF will be finalized automatically


@pytest.fixture
def source_grid_coords():
    """Create source grid coordinates (coarse resolution)."""
    src_lat = np.linspace(-90, 90, 46)  # 4-degree resolution
    src_lon = np.linspace(0, 360, 91, endpoint=False)  # 4-degree resolution
    return src_lat, src_lon


@pytest.fixture
def dest_grid_coords():
    """Create destination grid coordinates (fine resolution)."""
    dst_lat = np.linspace(-90, 90, 91)  # 2-degree resolution
    dst_lon = np.linspace(0, 360, 181, endpoint=False)  # 2-degree resolution
    return dst_lat, dst_lon


@pytest.fixture
def sample_data_2d(source_grid_coords):
    """Create sample 2D data for testing."""
    src_lat, src_lon = source_grid_coords
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)

    # Create a smooth function with interesting features
    data = (
        np.sin(np.radians(lat_2d) * 3)
        + np.cos(np.radians(lon_2d) * 2)
        + 0.5 * np.sin(np.radians(lat_2d) * 5) * np.cos(np.radians(lon_2d) * 3)
    )
    return data


class TestBilinearInterpolation:
    """Tests for bilinear interpolation method."""

    def test_bilinear_2d_data(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test bilinear interpolation on 2D data."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(sample_data_2d, opt)

        # Check output shape
        assert result.shape == (len(dst_lat), len(dst_lon))

        # Check data type
        assert result.dtype in [np.float32, np.float64]

        # Check that output is reasonable (not all NaN or zero)
        assert not np.all(np.isnan(result))
        assert not np.all(result == 0)

        # Check that values are in a reasonable range
        assert np.abs(result).max() < 10 * np.abs(sample_data_2d).max()

    def test_bilinear_with_poles(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test bilinear interpolation with pole handling."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.Pole = "all"

        result = ESMF_regrid(sample_data_2d, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_bilinear_3d_data(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test bilinear interpolation on 3D data (time, lat, lon)."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Create 3D data (time, lat, lon)
        n_time = 5
        data_3d = np.stack([sample_data_2d * (i + 1) for i in range(n_time)])

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data_3d, opt)

        # Check output shape
        assert result.shape == (n_time, len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_bilinear_4d_data(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test bilinear interpolation on 4D data (time, level, lat, lon)."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Create 4D data
        n_time = 3
        n_level = 4
        data_4d = np.stack(
            [[sample_data_2d * (i + 1) * (j + 1) for j in range(n_level)] for i in range(n_time)]
        )

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data_4d, opt)

        # Check output shape
        assert result.shape == (n_time, n_level, len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))


class TestPatchInterpolation:
    """Tests for patch recovery interpolation method."""

    def test_patch_2d_data(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test patch interpolation on 2D data."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "patch"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(sample_data_2d, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_patch_with_poles(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test patch interpolation with pole handling."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "patch"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.Pole = "all"

        result = ESMF_regrid(sample_data_2d, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))


class TestConservativeInterpolation:
    """Tests for conservative interpolation method."""

    def test_conserve_2d_data(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test conservative interpolation on 2D data."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "conserve"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(sample_data_2d, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_conserve_mass_conservation(self, source_grid_coords, dest_grid_coords):
        """Test that conservative method preserves mass (approximately)."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Create data with known total
        src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.ones_like(src_lat_2d)  # Uniform field

        opt = RegridOptions()
        opt.InterpMethod = "conserve"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.NormType = "FRACAREA"

        result = ESMF_regrid(data, opt)

        # Check that the result is close to uniform
        # (conservative interpolation should preserve constant fields)
        assert np.abs(result.mean() - 1.0) < 0.1


class TestNearestNeighborInterpolation:
    """Tests for nearest neighbor interpolation methods."""

    def test_neareststod_2d_data(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test nearest source to destination interpolation."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "neareststod"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(sample_data_2d, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_nearestdtos_2d_data(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test nearest destination to source interpolation."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "nearestdtos"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(sample_data_2d, opt)

        assert result.shape == (len(dst_lat), len(dst_lon))
        assert not np.all(np.isnan(result))

    def test_nearest_preserves_values(self, source_grid_coords, dest_grid_coords):
        """Test that nearest neighbor preserves exact values at matching points."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Create step function
        src_lon_2d, src_lat_2d = np.meshgrid(src_lon, src_lat)
        data = np.where(src_lat_2d > 0, 1.0, 0.0)

        opt = RegridOptions()
        opt.InterpMethod = "neareststod"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data, opt)

        # Check that result only contains 0 and 1
        unique_vals = np.unique(result[~np.isnan(result)])
        assert len(unique_vals) <= 2
        assert all(v in [0.0, 1.0] for v in unique_vals)


class TestInterpolationComparison:
    """Tests comparing different interpolation methods."""

    def test_smooth_function_bilinear_vs_patch(
        self, sample_data_2d, source_grid_coords, dest_grid_coords
    ):
        """Test that bilinear and patch give similar results for smooth functions."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Bilinear
        opt_bilinear = RegridOptions()
        opt_bilinear.InterpMethod = "bilinear"
        opt_bilinear.SrcGridLat = src_lat
        opt_bilinear.SrcGridLon = src_lon
        opt_bilinear.DstGridLat = dst_lat
        opt_bilinear.DstGridLon = dst_lon

        result_bilinear = ESMF_regrid(sample_data_2d, opt_bilinear)

        # Patch
        opt_patch = RegridOptions()
        opt_patch.InterpMethod = "patch"
        opt_patch.SrcGridLat = src_lat
        opt_patch.SrcGridLon = src_lon
        opt_patch.DstGridLat = dst_lat
        opt_patch.DstGridLon = dst_lon

        result_patch = ESMF_regrid(sample_data_2d, opt_patch)

        # Results should be reasonably close for smooth data
        valid = ~(np.isnan(result_bilinear) | np.isnan(result_patch))
        if valid.any():
            diff = np.abs(result_bilinear[valid] - result_patch[valid])
            assert np.mean(diff) < 1.0  # Reasonable tolerance


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_method(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test that invalid interpolation method raises error."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "invalid_method"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        with pytest.raises(ValueError, match="Invalid InterpMethod"):
            opt.validate()

    def test_missing_destination_coords(self, sample_data_2d, source_grid_coords):
        """Test that missing destination coordinates raises error."""
        src_lat, src_lon = source_grid_coords

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        # Destination coords not set

        with pytest.raises(ValueError, match="Destination"):
            opt.validate()

    def test_return_double(self, sample_data_2d, source_grid_coords, dest_grid_coords):
        """Test ReturnDouble option."""
        src_lat, src_lon = source_grid_coords
        dst_lat, dst_lon = dest_grid_coords

        # Input as float32
        data_float32 = sample_data_2d.astype(np.float32)

        opt = RegridOptions()
        opt.InterpMethod = "bilinear"
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon
        opt.ReturnDouble = True

        result = ESMF_regrid(data_float32, opt)

        assert result.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
