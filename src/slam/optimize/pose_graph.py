"""Sparse Gauss-Newton pose-graph optimization on SE(3)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from slam.geometry.lie import exp_se3, hat, invert, log_se3, orthonormalize


def adjoint(T: np.ndarray) -> np.ndarray:
    """Ad(T) for xi = [omega, v], so that T exp(xi) T^-1 = exp(Ad(T) xi)."""
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = A[3:, 3:] = R
    A[3:, :3] = hat(t) @ R
    return A


def _ad(xi: np.ndarray) -> np.ndarray:
    A = np.zeros((6, 6))
    A[:3, :3] = A[3:, 3:] = hat(xi[:3])
    A[3:, :3] = hat(xi[3:])
    return A


def pose_graph_se3(
    poses_cw: list[np.ndarray],
    edges: list[tuple[int, int, np.ndarray]],
    fixed: tuple[int, ...] = (0,),
    max_iter: int = 20,
) -> list[np.ndarray]:
    """edges: (i, j, T_ji) meaning T_j ≈ T_ji @ T_i (world-to-camera), identity information.

    Residual r = log(T_ji^-1 T_j T_i^-1); with left perturbations its Jacobians are
    -Jr^-1(r) for T_i and Jl^-1(r) Ad(T_ji^-1) for T_j (first-order Jr^-1, Jl^-1).
    """
    Ts = [orthonormalize(T) for T in poses_cw]
    free = [k for k in range(len(Ts)) if k not in fixed]
    fidx = np.full(len(Ts), -1)
    fidx[free] = np.arange(len(free))
    nf = len(free)
    if nf == 0 or not edges:
        return Ts

    eye = np.eye(6)
    for _ in range(max_iter):
        ab, blocks = [], []
        g = np.zeros(6 * nf)
        for i, j, T_ji in edges:
            Tm_inv = invert(T_ji)
            r = log_se3(Tm_inv @ Ts[j] @ invert(Ts[i]))
            half = 0.5 * _ad(r)
            jac = [(fidx[i], -(eye + half)), (fidx[j], (eye - half) @ adjoint(Tm_inv))]
            for a, Ja in jac:
                if a < 0:
                    continue
                g[6 * a : 6 * a + 6] += Ja.T @ r
                for b, Jb in jac:
                    if b >= 0:
                        ab.append((a, b))
                        blocks.append(Ja.T @ Jb)
        if not ab:
            break
        ab = np.asarray(ab)
        k = np.arange(6)
        rows = np.broadcast_to(6 * ab[:, 0, None, None] + k[None, :, None], (len(ab), 6, 6)).ravel()
        cols = np.broadcast_to(6 * ab[:, 1, None, None] + k[None, None, :], (len(ab), 6, 6)).ravel()
        H = sp.csc_matrix((np.asarray(blocks).ravel(), (rows, cols)), shape=(6 * nf,) * 2)
        dx = spla.spsolve(H + 1e-6 * sp.identity(6 * nf, format="csc"), -g).reshape(nf, 6)
        for idx, d in zip(free, dx):
            Ts[idx] = exp_se3(d) @ Ts[idx]
        if np.abs(dx).max() < 1e-7:
            break
    return Ts
