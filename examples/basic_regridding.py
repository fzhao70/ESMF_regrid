"""
Basic regridding examples using ESMF_regrid.

This example demonstrates basic usage of the ESMF_regrid function
with different interpolation methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from esmf_regrid import ESMF_regrid, RegridOptions


def example_bilinear_regridding():
    """Example of basic bilinear regridding."""
    print("=" * 60)
    print("Example 1: Bilinear Regridding")
    print("=" * 60)

    # Create source grid (4-degree resolution)
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)

    # Create destination grid (2-degree resolution)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create sample data with interesting features
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = (
        np.sin(np.radians(lat_2d) * 3)
        + np.cos(np.radians(lon_2d) * 2)
        + 0.5 * np.sin(np.radians(lat_2d) * 5) * np.cos(np.radians(lon_2d) * 3)
    )

    # Set up regridding options
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon

    # Perform regridding
    result = ESMF_regrid(data, opt)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
    print()

    return data, result, src_lat, src_lon, dst_lat, dst_lon


def example_conservative_regridding():
    """Example of conservative regridding."""
    print("=" * 60)
    print("Example 2: Conservative Regridding")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create uniform field to test conservation
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.ones_like(lat_2d)

    # Set up conservative regridding
    opt = RegridOptions()
    opt.InterpMethod = "conserve"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.NormType = "FRACAREA"

    # Perform regridding
    result = ESMF_regrid(data, opt)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input mean: {data.mean():.6f}")
    print(f"Output mean: {result.mean():.6f}")
    print(f"Difference: {abs(data.mean() - result.mean()):.6e}")
    print()

    return data, result


def example_nearest_neighbor():
    """Example of nearest neighbor regridding."""
    print("=" * 60)
    print("Example 3: Nearest Neighbor Regridding")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create categorical data (e.g., land cover types)
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.zeros_like(lat_2d)
    data[lat_2d > 30] = 1  # Type 1
    data[(lat_2d > 0) & (lat_2d <= 30)] = 2  # Type 2
    data[(lat_2d > -30) & (lat_2d <= 0)] = 3  # Type 3
    data[lat_2d <= -30] = 4  # Type 4

    # Set up nearest neighbor regridding
    opt = RegridOptions()
    opt.InterpMethod = "neareststod"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon

    # Perform regridding
    result = ESMF_regrid(data, opt)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Input unique values: {np.unique(data)}")
    print(f"Output unique values: {np.unique(result)}")
    print()

    return data, result


def example_multidimensional_data():
    """Example of regridding multi-dimensional data."""
    print("=" * 60)
    print("Example 4: Multi-dimensional Data Regridding")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create 4D data (time, level, lat, lon)
    n_time = 12  # months
    n_level = 5  # vertical levels

    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)

    # Create data varying in time and level
    data_4d = np.zeros((n_time, n_level, len(src_lat), len(src_lon)))
    for t in range(n_time):
        for lev in range(n_level):
            # Seasonal and vertical variation
            data_4d[t, lev] = (
                np.sin(np.radians(lat_2d) * 3)
                * np.cos(2 * np.pi * t / 12)  # Annual cycle
                * (1 + 0.2 * lev)  # Vertical variation
            )

    # Set up regridding
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon

    # Perform regridding
    result_4d = ESMF_regrid(data_4d, opt)

    print(f"Input shape: {data_4d.shape}")
    print(f"Output shape: {result_4d.shape}")
    print(f"Time steps: {n_time}")
    print(f"Vertical levels: {n_level}")
    print()

    return data_4d, result_4d


def example_grid_type_strings():
    """Example using grid type strings."""
    print("=" * 60)
    print("Example 5: Using Grid Type Strings")
    print("=" * 60)

    # Source data on 2-degree grid
    src_lat = np.linspace(-90, 90, 91)
    src_lon = np.linspace(0, 360, 181, endpoint=False)

    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

    # Regrid to 1-degree grid using string specification
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridType = "1deg"  # 1-degree global grid

    result = ESMF_regrid(data, opt)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Destination grid type: 1deg")
    print(f"Expected output shape: (181, 360)")
    print()

    return data, result


def compare_interpolation_methods():
    """Compare different interpolation methods."""
    print("=" * 60)
    print("Example 6: Comparing Interpolation Methods")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create test data
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

    methods = ["bilinear", "patch", "conserve", "neareststod"]
    results = {}

    for method in methods:
        opt = RegridOptions()
        opt.InterpMethod = method
        opt.SrcGridLat = src_lat
        opt.SrcGridLon = src_lon
        opt.DstGridLat = dst_lat
        opt.DstGridLon = dst_lon

        result = ESMF_regrid(data, opt)
        results[method] = result

        print(f"{method:12s}: mean={result.mean():7.4f}, "
              f"std={result.std():7.4f}, "
              f"range=[{result.min():7.4f}, {result.max():7.4f}]")

    print()
    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ESMF_regrid Basic Examples")
    print("=" * 60 + "\n")

    # Run examples
    example_bilinear_regridding()
    example_conservative_regridding()
    example_nearest_neighbor()
    example_multidimensional_data()
    example_grid_type_strings()
    compare_interpolation_methods()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
