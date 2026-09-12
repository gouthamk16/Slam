"""Closed-form SE(3) alignment (Horn) for 3D-3D matches."""

from __future__ import annotations

import numpy as np


def horn_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """T such that dst ≈ T @ src (points Nx3). Returns 4x4."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    X = src - cs
    Y = dst - cd
    S = X.T @ Y
    U, _, Vt = np.linalg.svd(S)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cd - R @ cs
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
