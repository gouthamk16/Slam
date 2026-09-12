"""Lie algebra helpers for SO(3) / SE(3).

Poses are 4x4 matrices. Tangent vectors are [omega (3), v (3)] with left
perturbation: T <- exp(xi) @ T.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64
    )


def vee(W: np.ndarray) -> np.ndarray:
    return np.array([W[2, 1], W[0, 2], W[1, 0]], dtype=np.float64)


def exp_so3(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta < EPS:
        return np.eye(3) + hat(w)
    k = w / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    cos_theta = 0.5 * (np.trace(R) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < EPS:
        return vee(0.5 * (R - R.T))
    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)


def _v_matrix(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = hat(w)
    if theta < EPS:
        return np.eye(3) + 0.5 * W
    return (
        np.eye(3)
        + (1.0 - np.cos(theta)) / (theta**2) * W
        + (theta - np.sin(theta)) / (theta**3) * (W @ W)
    )


def _v_inv_matrix(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    W = hat(w)
    if theta < EPS:
        return np.eye(3) - 0.5 * W
    a = 1.0 / (theta**2) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * W + a * (W @ W)


def exp_se3(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    w, v = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = exp_so3(w)
    T[:3, 3] = _v_matrix(w) @ v
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    w = log_so3(T[:3, :3])
    v = _v_inv_matrix(w) @ T[:3, 3]
    return np.concatenate([w, v])


def invert(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def orthonormalize(T: np.ndarray) -> np.ndarray:
    """Project the rotation block onto SO(3) (nearest rotation, via SVD)."""
    U, _, Vt = np.linalg.svd(np.asarray(T, dtype=np.float64)[:3, :3])
    out = np.array(T, dtype=np.float64)
    out[:3, :3] = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return out


def transform_points(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply SE(3) to Nx3 points."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    R, t = T[:3, :3], T[:3, 3]
    return X @ R.T + t
