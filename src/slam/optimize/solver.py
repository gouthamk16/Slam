"""Reprojection least squares: motion-only BA and sparse Schur-complement bundle adjustment.

Poses are world-to-camera T_cw with left perturbation T <- exp(xi) T, xi = [omega, v].
A measurement is (u, v, u_right); u_right is NaN for a monocular observation.
Huber kernel and chi2 outlier thresholds (95%, 2 or 3 dof) follow ORB-SLAM2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from slam.data.frame import StereoCalib
from slam.geometry.lie import exp_se3, orthonormalize

CHI2_MONO = 5.991
CHI2_STEREO = 7.815
_D3 = np.arange(3)
_D6 = np.arange(6)


@dataclass
class Obs:
    pose: np.ndarray   # (K,) pose index
    point: np.ndarray  # (K,) point index
    z: np.ndarray      # (K, 3) u, v, u_right
    info: np.ndarray   # (K,) 1 / sigma^2 of the keypoint octave


class _Lin(NamedTuple):
    r: np.ndarray       # (K, 3) residual pred - z
    J: np.ndarray       # (K, 3, 3) d(pred)/d(Xc)
    stereo: np.ndarray  # (K,)
    Xc: np.ndarray      # (K, 3)
    R: np.ndarray       # (K, 3, 3) rotation of each observation's pose


def _project(Xc: np.ndarray, z: np.ndarray, cam: StereoCalib):
    x, y, d = Xc.T
    iz = 1.0 / np.where(np.abs(d) > 1e-6, d, 1e-6)
    stereo = ~np.isnan(z[:, 2])
    pred = np.stack(
        [cam.fx * x * iz + cam.cx, cam.fy * y * iz + cam.cy, cam.fx * (x - cam.baseline) * iz + cam.cx],
        axis=1,
    )
    r = pred - z
    J = np.zeros((len(Xc), 3, 3))
    J[:, 0, 0] = J[:, 2, 0] = cam.fx * iz
    J[:, 1, 1] = cam.fy * iz
    J[:, 0, 2] = -cam.fx * x * iz**2
    J[:, 1, 2] = -cam.fy * y * iz**2
    J[:, 2, 2] = -cam.fx * (x - cam.baseline) * iz**2
    r[~stereo, 2] = 0.0
    J[~stereo, 2] = 0.0
    return r, J, stereo


def _dxi(Xc: np.ndarray) -> np.ndarray:
    """d(exp(xi) Xc)/d(xi) at xi = 0, i.e. [-[Xc]x | I], batched to (K, 3, 6)."""
    x, y, z = Xc.T
    D = np.zeros((len(Xc), 3, 6))
    D[:, 0, 1], D[:, 0, 2] = z, -y
    D[:, 1, 0], D[:, 1, 2] = -z, x
    D[:, 2, 0], D[:, 2, 1] = y, -x
    D[:, _D3, _D3 + 3] = 1.0
    return D


def _irls(r, stereo, depth, info, active, robust):
    """Per-observation weights, total (robust) cost, and chi2 inlier test."""
    chi2 = info * np.einsum("ki,ki->k", r, r)
    th = np.where(stereo, CHI2_STEREO, CHI2_MONO)
    w, rho = info.copy(), chi2.copy()
    if robust:
        s, d = np.sqrt(chi2), np.sqrt(th)
        big = s > d
        w[big] *= d[big] / s[big]
        rho[big] = 2.0 * d[big] * s[big] - th[big]
    ok = active & (depth > 0)
    w[~ok] = 0.0
    return w, float(rho[ok].sum()), (chi2 < th) & (depth > 0)


def motion_only_ba(
    T_cw: np.ndarray,
    Xw: np.ndarray,
    z: np.ndarray,
    info: np.ndarray,
    cam: StereoCalib,
    rounds: int = 4,
    iters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine one pose against fixed points. Outliers are re-classified after every round
    (and may come back); the last round runs without the robust kernel."""
    T = orthonormalize(T_cw)  # estimates stay on SE(3), as g2o's quaternions do
    active = np.ones(len(Xw), dtype=bool)
    for rnd in range(rounds):
        robust = rnd < rounds - 1
        for _ in range(iters):
            Xc = Xw @ T[:3, :3].T + T[:3, 3]
            r, Jpi, st = _project(Xc, z, cam)
            w, _, _ = _irls(r, st, Xc[:, 2], info, active, robust)
            J = Jpi @ _dxi(Xc)
            H = np.einsum("k,kai,kaj->ij", w, J, J)
            g = np.einsum("k,kai,ka->i", w, J, r)
            try:
                dx = np.linalg.solve(H + 1e-9 * np.eye(6), -g)
            except np.linalg.LinAlgError:
                return T, active
            T = exp_se3(dx) @ T
            if dx @ dx < 1e-12:
                break
        Xc = Xw @ T[:3, :3].T + T[:3, 3]
        r, _, st = _project(Xc, z, cam)
        _, _, active = _irls(r, st, Xc[:, 2], info, active, robust=False)
    return T, active


def _linearize(Ts: np.ndarray, X: np.ndarray, obs: Obs, cam: StereoCalib) -> _Lin:
    R = Ts[obs.pose, :3, :3]
    Xc = np.einsum("kij,kj->ki", R, X[obs.point]) + Ts[obs.pose, :3, 3]
    r, J, st = _project(Xc, obs.z, cam)
    return _Lin(r, J, st, Xc, R)


def _cost(lin: _Lin, obs: Obs, active: np.ndarray, robust: bool):
    return _irls(lin.r, lin.stereo, lin.Xc[:, 2], obs.info, active, robust)


def _schur_step(lin: _Lin, w, obs: Obs, fidx, n_pts: int, lam: float):
    """Solve the damped normal equations, eliminating points via the Schur complement."""
    Jx = lin.J @ lin.R
    wJx = w[:, None, None] * Jx
    Hxx = np.zeros((n_pts, 3, 3))
    gx = np.zeros((n_pts, 3))
    np.add.at(Hxx, obs.point, np.einsum("kai,kaj->kij", wJx, Jx))
    np.add.at(gx, obs.point, np.einsum("kai,ka->ki", wJx, lin.r))
    Hxx[:, _D3, _D3] = Hxx[:, _D3, _D3] * (1.0 + lam) + 1e-9
    Hxx_inv = np.linalg.inv(Hxx)

    nf = int(fidx.max()) + 1
    if nf == 0:
        return np.zeros((0, 6)), -np.einsum("qab,qb->qa", Hxx_inv, gx)

    f = fidx[obs.pose]
    m = f >= 0
    Jp = lin.J[m] @ _dxi(lin.Xc[m])
    wJp = w[m, None, None] * Jp
    Hpp = np.zeros((nf, 6, 6))
    gp = np.zeros((nf, 6))
    np.add.at(Hpp, f[m], np.einsum("kai,kaj->kij", wJp, Jp))
    np.add.at(gp, f[m], np.einsum("kai,ka->ki", wJp, lin.r[m]))
    Hpp[:, _D6, _D6] = Hpp[:, _D6, _D6] * (1.0 + lam) + 1e-9
    A = sp.bsr_matrix((Hpp, np.arange(nf), np.arange(nf + 1)), shape=(6 * nf,) * 2)

    Hpx = np.einsum("kai,kaj->kij", wJp, Jx[m])
    rows = np.broadcast_to(6 * f[m][:, None, None] + _D6[None, :, None], Hpx.shape).ravel()
    cols = np.broadcast_to(3 * obs.point[m][:, None, None] + _D3[None, None, :], Hpx.shape).ravel()
    B = sp.csr_matrix((Hpx.ravel(), (rows, cols)), shape=(6 * nf, 3 * n_pts))
    C = sp.bsr_matrix((Hxx_inv, np.arange(n_pts), np.arange(n_pts + 1)), shape=(3 * n_pts,) * 2)

    BC = B @ C
    S = A - BC @ B.T
    rhs = -gp.ravel() + BC @ gx.ravel()
    dp = np.linalg.solve(S.toarray(), rhs) if nf <= 200 else spla.spsolve(S.tocsc(), rhs)
    dx = C @ (-gx.ravel() - B.T @ dp)
    return dp.reshape(nf, 6), dx.reshape(n_pts, 3)


def bundle_adjust(
    poses: list[np.ndarray],
    fixed: np.ndarray,
    X: np.ndarray,
    obs: Obs,
    cam: StereoCalib,
    iters: tuple[int, ...] = (5, 10),
    stop: Callable[[], bool] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Levenberg-Marquardt over free poses and all points.

    Every stage but the last uses Huber; after each stage, observations failing chi2 are
    dropped for good (ORB-SLAM local BA). `stop` lets a caller abort between iterations.
    Returns (poses, points, inlier mask over obs).
    """
    Ts = np.array([orthonormalize(T) for T in poses])
    X = np.array(X, dtype=np.float64)
    free = np.flatnonzero(~np.asarray(fixed, dtype=bool))
    fidx = np.full(len(Ts), -1)
    fidx[free] = np.arange(len(free))
    active = np.ones(len(obs.pose), dtype=bool)
    lin = _linearize(Ts, X, obs, cam)

    for stage, n_iter in enumerate(iters):
        robust = stage < len(iters) - 1
        lam = 1e-4
        w, cost, _ = _cost(lin, obs, active, robust)
        for _ in range(n_iter):
            if stop is not None and stop():
                break
            dp, dx = _schur_step(lin, w, obs, fidx, len(X), lam)
            Ts_new = Ts.copy()
            for k, i in enumerate(free):
                Ts_new[i] = exp_se3(dp[k]) @ Ts[i]
            X_new = X + dx
            lin_new = _linearize(Ts_new, X_new, obs, cam)
            w_new, cost_new, _ = _cost(lin_new, obs, active, robust)
            if cost_new < cost:
                done = cost - cost_new < 1e-6 * cost
                Ts, X, lin, w, cost = Ts_new, X_new, lin_new, w_new, cost_new
                lam = max(lam / 3.0, 1e-8)
                if done:
                    break
            else:
                lam *= 10.0
                if lam > 1e6:
                    break
        _, _, good = _cost(lin, obs, active, robust=False)
        active &= good
        if stop is not None and stop():
            break
    return list(Ts), X, active
