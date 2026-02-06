import numpy as np
import open3d as o3d
from tqdm import trange

from .utils import norm_m1p1

def label_precipitation(
    points: np.ndarray,
    intensities: np.ndarray,
    *,
    k: int = 50,
    a: float = 1,
    b: float = 10,
    c: float = 1,
    th_md: float = 0.5,
    th_s: float = 0.75,
    th_ns: float = -0.75,
    sor_k: int = 10,
    sor_ratio: float = 10,
    th_d: float | None = None,
    print_progress: bool = False,
) -> np.ndarray:
    """
    Label precipitation / snowflake outliers in a 3D point cloud.

    Notes
    -----
    Statistical Outlier Removal (SOR) is used as an initial step.
    Points removed by SOR are also labeled as outliers in the final result.

    Returns
    -------
    labels : (N,) ndarray
        0 = precipitation / outlier
        1 = non-precipitation / inlier
    """

    # --- Open3D point cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # --- Statistical Outlier Removal ---
    pcd_filt, ind = pcd.remove_statistical_outlier(
        nb_neighbors=sor_k,
        std_ratio=sor_ratio
    )

    pts = np.asarray(pcd_filt.points)          # <-- cached ONCE
    intensities_filt = intensities[ind]

    # --- KD-tree ---
    kdtree = o3d.geometry.KDTreeFlann(pcd_filt)

    eigenvalues = []
    knn_md = []
    knn_idx_list = []
    knn_dist_list = []

    iterator = trange(len(pts), disable=not print_progress, desc="Computing features")

    for i in iterator:
        _, idx, distances = kdtree.search_knn_vector_3d(pts[i], k)

        idx = np.asarray(idx)
        distances = np.asarray(distances)

        knn_idx_list.append(idx)
        knn_dist_list.append(distances)

        neighbors = pts[idx]

        # mean kNN distance normalized by range
        knn_md.append(np.mean(distances) / np.linalg.norm(pts[i]))

        # PCA via covariance
        cov = np.cov(neighbors.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigenvalues.append(np.sort(eigvals)[::-1])

    eigenvalues = np.asarray(eigenvalues)
    knn_md = np.clip(knn_md, None, th_md)

    # variance ratio
    varR = eigenvalues[:, 2] / np.sum(eigenvalues, axis=1)

    # feature definition
    features = (1 / (intensities_filt + a)) * (knn_md + b) * (varR + c)
    features_norm = norm_m1p1(features)

    # --- Initial labeling ---
    feat_labels = np.zeros(len(features_norm))
    feat_labels[features_norm < th_ns] = -1
    feat_labels[features_norm > th_s] = 1

    # --- Label propagation for undecided points ---
    idx_tbd = np.where(feat_labels == 0)[0]
    yi_list = []

    for idx in idx_tbd:
        yj = feat_labels[knn_idx_list[idx]]
        dij = knn_dist_list[idx]
        mask = dij > 0
        yi = np.sum(yj[mask] / dij[mask]) + yj[0]
        yi_list.append(yi)

    feat_labels[idx_tbd] = np.where(np.asarray(yi_list) > 0, 1, -1)

    # convert to 0/1 convention
    labels_filt = np.where(feat_labels < 0, 1, 0)

    # --- Reinsert SOR-removed points ---
    labels = np.zeros(len(points))
    labels[ind] = labels_filt

    # --- Optional distance threshold ---
    if th_d is not None:
        d = np.linalg.norm(points, axis=1)
        labels[d > th_d] = 0

    return labels
