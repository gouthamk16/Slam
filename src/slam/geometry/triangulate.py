"""Linear two-view triangulation."""

from __future__ import annotations

import numpy as np


def triangulate_dlt(T1_cw: np.ndarray, T2_cw: np.ndarray, xn1: np.ndarray, xn2: np.ndarray) -> np.ndarray:
    """Batched DLT from normalized image coordinates (M, 2). NaN rows where X is at infinity."""
    P1, P2 = T1_cw[:3], T2_cw[:3]
    A = np.stack(
        [
            xn1[:, 0, None] * P1[2] - P1[0],
            xn1[:, 1, None] * P1[2] - P1[1],
            xn2[:, 0, None] * P2[2] - P2[0],
            xn2[:, 1, None] * P2[2] - P2[1],
        ],
        axis=1,
    )
    V = np.linalg.svd(A)[2][:, -1]
    w = np.where(np.abs(V[:, 3]) > 1e-12, V[:, 3], np.nan)
    return V[:, :3] / w[:, None]
