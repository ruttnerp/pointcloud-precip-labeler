import numpy as np


def norm_m1p1(x: np.ndarray) -> np.ndarray:
    """
    Normalize array to the range [-1, 1] using min-max normalization.
    """
    x = np.asarray(x)
    x_min = np.min(x)
    x_max = np.max(x)

    if x_max == x_min:
        return np.zeros_like(x)

    return -1.0 + 2.0 * (x - x_min) / (x_max - x_min)


def add_precip_label(point_data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Append precipitation / outlier label as last column to point_data.

    Expected format:
    [timestamp, x, y, z, intensity, ...]
    """
    from .label_precipitation import label_precipitation

    if point_data.shape[0] == 0:
        return point_data

    if point_data.shape[0] <= 3:
        labels = np.ones(point_data.shape[0])
    else:
        points = point_data[:, 1:4]
        intensities = point_data[:, 4]
        labels = label_precipitation(points, intensities, **kwargs)

    return np.hstack((point_data, labels[:, None]))
