# ESMF_regrid

A Python implementation of NCL's ESMF_regrid function using ESMPy.

## Overview

`ESMF_regrid` is a fast regridding library that provides functionality similar to NCL's ESMF_regrid function. It supports regridding between rectilinear, curvilinear, and unstructured grids using the Earth System Modeling Framework (ESMF).

## Features

- **Multiple Interpolation Methods**:
  - Bilinear
  - Patch (higher-order patch recovery)
  - Conservative (first-order and second-order)
  - Nearest neighbor (source-to-destination and destination-to-source)

- **Grid Types**:
  - Rectilinear grids (1D latitude/longitude)
  - Curvilinear grids (2D latitude/longitude)
  - Unstructured grids (mesh)
  - Regional and global grids

- **Advanced Features**:
  - Masking support for source and destination grids
  - Missing value handling
  - Pole handling for periodic grids
  - Multi-dimensional data support (time, level, lat, lon)
  - Weight file caching and reuse
  - Extrapolation methods

## Installation

### Prerequisites

1. Install ESMF library (required for ESMPy):
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install libesmf-dev

   # On macOS with Homebrew
   brew install esmf

   # Or build from source
   # See: https://earthsystemmodeling.org/docs/release/latest/ESMF_usrdoc/
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

   Or install dependencies separately:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage

```python
import numpy as np
from esmf_regrid import ESMF_regrid, RegridOptions

# Create source grid (4-degree resolution)
src_lat = np.linspace(-90, 90, 46)
src_lon = np.linspace(0, 360, 91, endpoint=False)

# Create destination grid (2-degree resolution)
dst_lat = np.linspace(-90, 90, 91)
dst_lon = np.linspace(0, 360, 181, endpoint=False)

# Create sample data
lon_2d, lat_2d = np.meshgrid(src_lon, src_lat)
data = np.sin(np.radians(lat_2d) * 3) + np.cos(np.radians(lon_2d) * 2)

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
```

### Using Grid Type Strings

```python
from esmf_regrid import ESMF_regrid, RegridOptions

# Your data on a custom grid
opt = RegridOptions()
opt.InterpMethod = "bilinear"
opt.SrcGridLat = src_lat
opt.SrcGridLon = src_lon
opt.DstGridType = "1deg"  # Destination is 1-degree global grid

result = ESMF_regrid(data, opt)
```

### Conservative Regridding

```python
opt = RegridOptions()
opt.InterpMethod = "conserve"
opt.SrcGridLat = src_lat
opt.SrcGridLon = src_lon
opt.DstGridLat = dst_lat
opt.DstGridLon = dst_lon
opt.NormType = "FRACAREA"  # Normalization type

result = ESMF_regrid(data, opt)
```

### Regridding with Masking

```python
# Create a mask (1 = masked, 0 = valid)
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
```

### Multi-dimensional Data

```python
# 4D data: (time, level, lat, lon)
data_4d = np.random.randn(10, 5, len(src_lat), len(src_lon))

opt = RegridOptions()
opt.InterpMethod = "bilinear"
opt.SrcGridLat = src_lat
opt.SrcGridLon = src_lon
opt.DstGridLat = dst_lat
opt.DstGridLon = dst_lon

# Automatically handles extra dimensions
result_4d = ESMF_regrid(data_4d, opt)
print(result_4d.shape)  # (10, 5, len(dst_lat), len(dst_lon))
```

## RegridOptions Reference

### File Names
- `SrcFileName`: Source grid file name (default: "source_grid_file.nc")
- `DstFileName`: Destination grid file name (default: "destination_grid_file.nc")
- `WgtFileName`: Weight file name (default: "weights_file.nc")

### Interpolation Method
- `InterpMethod`: Interpolation method (default: "bilinear")
  - Options: "bilinear", "patch", "conserve", "neareststod", "nearestdtos"

### Grid Coordinates
- `SrcGridLat`, `SrcGridLon`: Source grid coordinates (1D or 2D arrays)
- `DstGridLat`, `DstGridLon`: Destination grid coordinates (1D or 2D arrays)
- `SrcGridType`, `DstGridType`: Grid type strings (e.g., "1deg", "0.25deg", "2x2")

### Masking
- `SrcGridMask`, `DstGridMask`: Mask arrays (1 = masked, 0 = valid)
- `SrcMissingValue`, `DstMissingValue`: Missing value indicators

### Regional Grids
- `SrcRegional`, `DstRegional`: Whether grids are regional (default: False)

### Advanced Options
- `Pole`: Pole handling ("none", "all", "teeth") (default: "all")
- `LineType`: Distance calculation ("GREAT_CIRCLE", "CART") (default: "GREAT_CIRCLE")
- `NormType`: Normalization for conservative regridding (default: "FRACAREA")
- `IgnoreUnmappedPoints`: Ignore unmapped destination points (default: False)
- `ExtrapMethod`: Extrapolation method ("neareststod", "nearestidavg", "creep")
- `ReturnDouble`: Return double precision (default: False)
- `ForceOverwrite`: Force weight file regeneration (default: False)

## Examples

See the `examples/` directory for more comprehensive examples:

- `examples/basic_regridding.py`: Basic regridding examples
- `examples/regional_grids.py`: Working with regional grids
- `examples/curvilinear_grids.py`: Curvilinear grid regridding
- `examples/masking_examples.py`: Using masks and missing values
- `examples/multidimensional_data.py`: Handling multi-dimensional data

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test modules:

```bash
pytest tests/test_interpolation_methods.py -v
pytest tests/test_grid_types.py -v
pytest tests/test_masking.py -v
```

## Performance Tips

1. **Reuse Weight Files**: If regridding multiple variables with the same grids, reuse the weight file:
   ```python
   opt.WgtFileName = "my_weights.nc"
   opt.ForceOverwrite = False  # Use existing weights if available
   ```

2. **Use Conservative Method for Flux Variables**: For variables like precipitation or radiation, use conservative regridding to preserve totals.

3. **Batch Processing**: For multiple time steps, pass all data at once as a 3D/4D array rather than looping.

## Differences from NCL's ESMF_regrid

- This implementation uses ESMPy directly for better performance
- Some advanced ESMF options may have different names
- Weight file format is ESMF native format
- Automatic coordinate detection works slightly differently

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- [NCL ESMF_regrid Documentation](https://www.ncl.ucar.edu/Document/Functions/ESMF/ESMF_regrid.shtml)
- [ESMF Documentation](https://earthsystemmodeling.org/docs/)
- [ESMPy Documentation](https://earthsystemmodeling.org/esmpy/)

## Citation

If you use this library in your research, please cite:

```
ESMF_regrid: A Python implementation of NCL's ESMF_regrid function
```

## Support

For issues and questions:
- GitHub Issues: [https://github.com/yourusername/esmf_regrid/issues](https://github.com/yourusername/esmf_regrid/issues)
- ESMF Forum: [https://earthsystemmodeling.org/support/](https://earthsystemmodeling.org/support/)
