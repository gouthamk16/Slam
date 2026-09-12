"""Trajectory metrics: SE(3)-aligned ATE and KITTI-devkit-style RPE."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from slam.geometry.align import horn_se3
from slam.geometry.lie import invert, log_so3


def centers(traj_cw: list[np.ndarray]) -> np.ndarray:
    return np.stack([invert(T)[:3, 3] for T in traj_cw])


def ate_rmse(traj_cw: list[np.ndarray], gt_cw: list[np.ndarray]) -> float:
    n = min(len(traj_cw), len(gt_cw))
    est, gt = centers(traj_cw[:n]), centers(gt_cw[:n])
    T = horn_se3(est, gt)
    aligned = est @ T[:3, :3].T + T[:3, 3]
    return float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1))))


def kitti_rpe(
    traj_cw: list[np.ndarray],
    gt_cw: list[np.ndarray],
    lengths: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800),
    step: int = 10,
) -> tuple[float, float]:
    """Mean drift over path segments: translation in %, rotation in deg/100 m."""
    n = min(len(traj_cw), len(gt_cw))
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centers(gt_cw[:n]), axis=0), axis=1))])
    t_err, r_err = [], []
    for i in range(0, n, step):
        for L in lengths:
            j = int(np.searchsorted(dist, dist[i] + L))
            if j >= n:
                continue
            E = invert(traj_cw[j] @ invert(traj_cw[i])) @ (gt_cw[j] @ invert(gt_cw[i]))
            t_err.append(np.linalg.norm(E[:3, 3]) / L)
            r_err.append(np.degrees(np.linalg.norm(log_so3(E[:3, :3]))) / L)
    if not t_err:
        return float("nan"), float("nan")
    return 100.0 * float(np.mean(t_err)), 100.0 * float(np.mean(r_err))


def write_kitti_poses(path: Path, traj_cw: list[np.ndarray]) -> None:
    rows = [invert(T)[:3, :4].reshape(-1) for T in traj_cw]
    np.savetxt(path, np.asarray(rows), fmt="%.6e")
