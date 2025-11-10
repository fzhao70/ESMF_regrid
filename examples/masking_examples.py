"""
Masking examples for ESMF_regrid.

This example demonstrates how to use masks and handle missing values
when regridding data.
"""

import numpy as np
from esmf_regrid import ESMF_regrid, RegridOptions


def example_source_masking():
    """Example of masking source grid points."""
    print("=" * 60)
    print("Example 1: Source Grid Masking")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create sample data
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

    # Create land-sea mask (1 = masked/ocean, 0 = valid/land)
    # Mask western hemisphere as "ocean"
    src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
    src_mask[:, src_lon < 180] = 1

    # Regrid with mask
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.SrcGridMask = src_mask
    opt.IgnoreUnmappedPoints = True

    result = ESMF_regrid(data, opt)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Masked points in source: {src_mask.sum()}")
    print(f"Valid output points: {(~np.isnan(result)).sum()}")
    print()


def example_destination_masking():
    """Example of masking destination grid points."""
    print("=" * 60)
    print("Example 2: Destination Grid Masking")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create sample data
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

    # Create destination mask (mask polar regions)
    dst_mask = np.zeros((len(dst_lat), len(dst_lon)), dtype=np.int32)
    dst_mask[dst_lat < -60, :] = 1  # Mask Antarctica
    dst_mask[dst_lat > 60, :] = 1  # Mask Arctic

    # Regrid with destination mask
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.DstGridMask = dst_mask
    opt.IgnoreUnmappedPoints = True

    result = ESMF_regrid(data, opt)

    print(f"Input shape: {data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Masked points in destination: {dst_mask.sum()}")
    print()


def example_missing_values():
    """Example of handling missing values in data."""
    print("=" * 60)
    print("Example 3: Missing Values in Data")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create sample data
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

    # Add missing values (e.g., over mountains or bad observations)
    data[10:15, 20:30] = np.nan
    data[30:35, 50:60] = np.nan

    print(f"Missing points in source: {np.isnan(data).sum()}")

    # Regrid with missing values
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.IgnoreUnmappedPoints = True

    result = ESMF_regrid(data, opt)

    print(f"Output shape: {result.shape}")
    print(f"Missing points in output: {np.isnan(result).sum()}")
    print()


def example_specific_missing_value():
    """Example using a specific missing value indicator."""
    print("=" * 60)
    print("Example 4: Specific Missing Value Indicator")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create sample data
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

    # Use specific missing value (common in climate data)
    missing_val = -999.0
    data[10:15, 20:30] = missing_val
    data[30:35, 50:60] = missing_val

    print(f"Missing value indicator: {missing_val}")
    print(f"Points with missing value: {(data == missing_val).sum()}")

    # Regrid with specific missing value
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.SrcMissingValue = missing_val
    opt.IgnoreUnmappedPoints = True

    result = ESMF_regrid(data, opt)

    print(f"Output shape: {result.shape}")
    print()


def example_combined_mask_and_missing():
    """Example with both mask and missing values."""
    print("=" * 60)
    print("Example 5: Combined Mask and Missing Values")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create sample data
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

    # Add missing values
    data[5:10, 10:20] = np.nan

    # Add mask (e.g., land-sea mask)
    src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)
    src_mask[:, :30] = 1  # Mask first 30 longitude points

    print(f"Missing points in data: {np.isnan(data).sum()}")
    print(f"Masked points: {src_mask.sum()}")

    # Regrid with both mask and missing values
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.SrcGridMask = src_mask
    opt.IgnoreUnmappedPoints = True

    result = ESMF_regrid(data, opt)

    print(f"Output shape: {result.shape}")
    print()


def example_regional_ocean_mask():
    """Example of masking for ocean-only data."""
    print("=" * 60)
    print("Example 6: Regional Ocean Masking")
    print("=" * 60)

    # Create grids for Pacific Ocean region
    src_lat = np.linspace(-30, 30, 31)  # Tropical Pacific
    src_lon = np.linspace(120, 280, 81)  # 120E to 80W
    dst_lat = np.linspace(-30, 30, 61)
    dst_lon = np.linspace(120, 280, 161)

    # Create sample data (e.g., sea surface temperature)
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = 25 + 5 * np.sin(np.radians(lat_2d) * 2) * np.cos(np.radians(lon_2d - 180) * 1.5)

    # Create simplified land mask
    # This is a very simplified example - real land masks would be more complex
    src_mask = np.zeros((len(src_lat), len(src_lon)), dtype=np.int32)

    # Mask some "land" areas (simplified)
    # Eastern Pacific coast
    src_mask[:, src_lon > 260] = 1
    # Maritime continent
    src_mask[:, (src_lon > 120) & (src_lon < 140)] = 1

    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Ocean points: {(src_mask == 0).sum()}")
    print(f"Land points (masked): {(src_mask == 1).sum()}")

    # Regrid ocean data
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.SrcGridMask = src_mask
    opt.SrcRegional = True
    opt.DstRegional = True
    opt.IgnoreUnmappedPoints = True

    result = ESMF_regrid(data, opt)

    print(f"Output shape: {result.shape}")
    print(f"Output range: [{np.nanmin(result):.2f}, {np.nanmax(result):.2f}]")
    print()


def example_polar_region_handling():
    """Example of handling polar regions with masks."""
    print("=" * 60)
    print("Example 7: Polar Region Handling")
    print("=" * 60)

    # Create grids
    src_lat = np.linspace(-90, 90, 46)
    src_lon = np.linspace(0, 360, 91, endpoint=False)
    dst_lat = np.linspace(-90, 90, 91)
    dst_lon = np.linspace(0, 360, 181, endpoint=False)

    # Create sample data
    lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
    data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

    # Mask data at poles (common for some satellite observations)
    data[np.abs(lat_2d) > 85] = np.nan

    print(f"Valid data coverage: {(~np.isnan(data)).sum()} / {data.size} points")

    # Regrid with pole options
    opt = RegridOptions()
    opt.InterpMethod = "bilinear"
    opt.SrcGridLat = src_lat
    opt.SrcGridLon = src_lon
    opt.DstGridLat = dst_lat
    opt.DstGridLon = dst_lon
    opt.Pole = "none"  # Don't use pole averaging
    opt.IgnoreUnmappedPoints = True

    result = ESMF_regrid(data, opt)

    print(f"Output shape: {result.shape}")
    print(f"Valid output points: {(~np.isnan(result)).sum()} / {result.size}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ESMF_regrid Masking Examples")
    print("=" * 60 + "\n")

    # Run examples
    example_source_masking()
    example_destination_masking()
    example_missing_values()
    example_specific_missing_value()
    example_combined_mask_and_missing()
    example_regional_ocean_mask()
    example_polar_region_handling()

    print("=" * 60)
    print("All masking examples completed successfully!")
    print("=" * 60)
