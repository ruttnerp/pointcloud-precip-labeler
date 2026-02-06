# pointcloud-precip-labeler

A Python utility for labeling precipitation (e.g. snowflakes)
and sparse outliers in 3D point cloud data.

The method combines geometric neighborhood analysis with intensity-based
features and is designed for lidar-style point clouds.

---

## Features
- Labels precipitation / sparse outliers in raw XYZ + intensity point clouds
- Uses local PCA geometry and kNN statistics
- Includes statistical outlier removal (SOR)
- Minimal dependencies
- Designed for research use and easy modification

---

## Installation

### Recommended (Conda / Miniforge)

```bash
conda env create -f environment.yml
conda activate pointcloud-precip
```

### Alternative (pip)
```bash
pip install -r requirements.txt
```

## Usage
```python
from pointcloud_precip_labeler import add_precip_label
point_data_labeled = add_precip_label(point_data, th_d=250)
```

The precipitation / outlier label is appended as the last column.
See `examples/minimal_example.py` for a complete runnable example.

## Input format

Expected `point_data` layout:

| Column Index | Description                    |
|--------------|--------------------------------|
| 0            | Timestamp                       |
| 1            | X coordinate                    |
| 2            | Y coordinate                    |
| 3            | Z coordinate                    |
| 4            | Intensity                       |
| 5+           | Optional metadata               |

## Output

- Label `0`: precipitation / outlier
- Label `1`: non-precipitation / inlier

Points removed during statistical outlier removal (SOR) are also labeled as outliers in the final output.

## Notes

- This code is intended for research and experimentation.
- Parameters are exposed to allow adaptation to different sensors and environments.

## License

This project is licensed under the MIT License.