import numpy as np
from pointcloud_precip_labeler import add_precip_label


if __name__ == "__main__":
    n = 2000

    timestamps = np.random.uniform(0, 1, n)
    xyz = np.random.randn(n, 3)
    intensities = np.random.uniform(0, 1, n)
    tag = np.zeros(n)
    return_nr = np.ones(n)

    point_data = np.column_stack(
        (timestamps, xyz, intensities, tag, return_nr)
    )

    point_data_labeled = add_precip_label(point_data, th_d=250)

    labels = point_data_labeled[:, -1]

    print(f"Total points: {len(labels)}")
    print(f"Outliers (precipitation): {(labels == 0).sum()}")
    print(f"Inliers: {(labels == 1).sum()}")
