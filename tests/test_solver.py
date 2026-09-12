import numpy as np

from slam.data.frame import StereoCalib
from slam.geometry.lie import exp_se3, invert, log_se3
from slam.optimize.pose_graph import pose_graph_se3
from slam.optimize.solver import Obs, bundle_adjust, motion_only_ba

CAM = StereoCalib(718.0, 718.0, 607.0, 185.0, 0.54, 1241, 376)


def _scene(rng, n):
    return np.column_stack([rng.uniform(-10, 10, n), rng.uniform(-2, 2, n), rng.uniform(5, 40, n)])


def _measure(T, X, rng, noise):
    z = CAM.project_stereo(X @ T[:3, :3].T + T[:3, 3]) + rng.normal(0, noise, (len(X), 3))
    z[rng.random(len(X)) < 0.3, 2] = np.nan
    return z


def _corrupt(z, bad, rng):
    shift = rng.uniform(20, 60, (int(bad.sum()), 2))
    z[bad, 0] += shift[:, 0]
    z[bad, 1] += shift[:, 1]
    z[bad, 2] += shift[:, 0]


def _pose_err(T, T_true):
    return np.linalg.norm(log_se3(T @ invert(T_true)))


def test_motion_only_ba_rejects_outliers():
    rng = np.random.default_rng(0)
    X = _scene(rng, 300)
    T_true = exp_se3([0.02, -0.05, 0.01, 0.3, -0.1, 1.2])
    z = _measure(T_true, X, rng, 0.5)
    bad = rng.random(len(X)) < 0.2
    _corrupt(z, bad, rng)
    T0 = exp_se3([0.03, 0.02, -0.02, 0.4, 0.2, -0.5]) @ T_true
    T, inl = motion_only_ba(T0, X, z, np.ones(len(X)), CAM)
    assert _pose_err(T, T_true) < 0.01
    assert (inl == ~bad).mean() > 0.97


def _orth_err(T):
    R = T[:3, :3]
    return np.abs(R.T @ R - np.eye(3)).max()


def test_optimizers_return_rigid_poses():
    """Constant-velocity prediction amplifies any non-orthogonality ~3x per frame,
    so every estimator must hand back poses on SE(3), as g2o's quaternions do."""
    rng = np.random.default_rng(2)
    X = _scene(rng, 200)
    T_true = exp_se3([0.01, 0.02, 0.0, 0.1, 0.0, 0.8])
    z = _measure(T_true, X, rng, 0.3)
    skew = T_true.copy()
    skew[:3, :3] *= 1.0 + 1e-4  # a pose that already drifted off SO(3)

    T, _ = motion_only_ba(skew, X, z, np.ones(len(X)), CAM)
    assert _orth_err(T) < 1e-12

    obs = Obs(np.zeros(len(X), dtype=int), np.arange(len(X)), z, np.ones(len(X)))
    P, _, _ = bundle_adjust([skew], np.array([False]), X, obs, CAM, iters=(2,))
    assert _orth_err(P[0]) < 1e-12

    out = pose_graph_se3([np.eye(4), skew], [(0, 1, T_true)], fixed=(0,))
    assert _orth_err(out[1]) < 1e-12


def test_bundle_adjust_recovers_poses_and_points():
    rng = np.random.default_rng(1)
    X = _scene(rng, 400)
    poses = [exp_se3([0, 0.02 * k, 0, 0, 0, -1.0 * k]) for k in range(6)]
    pi, qi, zs = [], [], []
    for k, T in enumerate(poses):
        z = _measure(T, X, rng, 0.2)
        vis = ((X @ T[:3, :3].T + T[:3, 3])[:, 2] > 1) & CAM.in_image(z[:, :2])
        pi += [k] * int(vis.sum())
        qi += list(np.flatnonzero(vis))
        zs.append(z[vis])
    obs = Obs(np.array(pi), np.array(qi), np.vstack(zs), np.ones(len(pi)))
    bad = rng.random(len(pi)) < 0.05
    _corrupt(obs.z, bad, rng)

    noisy = poses[:2] + [exp_se3(rng.normal(0, [0.01] * 3 + [0.2] * 3)) @ T for T in poses[2:]]
    X0 = X + rng.normal(0, 0.3, X.shape)
    fixed = np.array([True, True, False, False, False, False])
    P, Xo, inl = bundle_adjust(noisy, fixed, X0, obs, CAM)

    assert max(_pose_err(T, Tt) for T, Tt in zip(P, poses)) < 0.02
    seen = np.unique(obs.point)
    assert np.median(np.linalg.norm(Xo[seen] - X[seen], axis=1)) < 0.25
    assert inl[bad].mean() < 0.1 and inl[~bad].mean() > 0.95
