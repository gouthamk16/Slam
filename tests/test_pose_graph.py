import numpy as np

from slam.geometry.lie import exp_se3, invert
from slam.optimize.pose_graph import pose_graph_se3


def test_pose_graph_recovers_relative_pose():
    T0 = np.eye(4)
    T1_true = exp_se3([0.0, 0.05, 0.1, 1.2, 0.0, 0.3])
    T1_noisy = exp_se3([0.1, -0.05, 0.0, 0.4, 0.2, -0.2]) @ T1_true
    T_rel = T1_true @ invert(T0)
    out = pose_graph_se3([T0, T1_noisy], [(0, 1, T_rel)], fixed=(0,), max_iter=25)
    err = np.linalg.norm(out[1] - T1_true)
    assert err < 0.05


def test_pose_graph_closes_noisy_loop():
    rng = np.random.default_rng(0)
    n = 30

    def T_wc(k):
        th = 2 * np.pi * k / n
        T = exp_se3([0, th, 0, 0, 0, 0])
        T[:3, 3] = [10 * np.cos(th), 0, 10 * np.sin(th)]
        return T

    gt = [invert(T_wc(k)) for k in range(n)]
    est, edges = [gt[0]], []
    for k in range(n - 1):
        odo = exp_se3(rng.normal(0, [0.01] * 3 + [0.05] * 3)) @ gt[k + 1] @ invert(gt[k])
        edges.append((k, k + 1, odo))
        est.append(odo @ est[-1])
    edges.append((n - 1, 0, gt[0] @ invert(gt[n - 1])))
    out = pose_graph_se3(est, edges, fixed=(0,))

    def err(P):
        return max(np.linalg.norm(invert(a)[:3, 3] - invert(b)[:3, 3]) for a, b in zip(P, gt))

    assert err(est) > 1.0
    assert err(out) < 0.3 * err(est)
