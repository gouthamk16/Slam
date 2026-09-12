"""Local mapping: keyframe insertion, point culling and creation, fusion, local BA, keyframe culling.

Follows ORB-SLAM2's LocalMapping for stereo input.
"""

from __future__ import annotations

import numpy as np

from slam.features.matcher import TH_LOW, best_two, hamming, one_to_one, rotation_consistent, same_group_pairs, window_pairs
from slam.features.orb import INV_SIGMA2, SCALE, SCALES
from slam.geometry.lie import hat, transform_points
from slam.geometry.triangulate import triangulate_dlt
from slam.map.structure import KeyFrame, Map, MapPoint, point_arrays, update_normals_and_depths, visible
from slam.optimize.solver import CHI2_MONO, CHI2_STEREO, Obs, bundle_adjust
from slam.tracking import stereo_depth, unproject
from slam.worker import Worker


class LocalMapper(Worker):
    def __init__(self, map_: Map, vocab=None, th_depth: float = 35.0, threaded: bool = False):
        super().__init__(threaded)
        self.map = map_
        self.vocab = vocab
        self.th_depth = th_depth
        self.recent: list[MapPoint] = []
        self.loop = None
        self.abort = False

    # --- interface used by the tracker ---
    def insert(self, kf: KeyFrame) -> None:
        self.abort = True
        super().insert(kf)

    def idle(self) -> bool:
        return not self.busy and not self.queue

    def queue_len(self) -> int:
        return len(self.queue)

    def interrupt_ba(self) -> None:
        self.abort = True

    # --- keyframe processing ---
    def process(self, kf: KeyFrame) -> None:
        with self.map.lock:
            self._add_keyframe(kf)
            self._cull_points(kf)
            self._triangulate(kf)
            if not self.queue:
                self._fuse_neighbors(kf)
        self.abort = False
        if not self.queue and len(self.map.keyframes) > 2:
            self._local_ba(kf)
        with self.map.lock:
            self._cull_keyframes(kf)
        if self.loop is not None:
            self.loop.insert(kf)

    def _add_keyframe(self, kf: KeyFrame) -> None:
        if self.vocab is not None:
            self.vocab.compute(kf)
        for i, mp in kf.valid_points():
            if kf in mp.obs:
                self.recent.append(mp)  # created by tracking for this keyframe
            else:
                mp.add_observation(kf, i)
                mp.update_normal_and_depth()
                mp.update_descriptor()
        kf.update_connections()

    def _cull_points(self, kf: KeyFrame) -> None:
        keep = []
        for mp in self.recent:
            if mp.bad:
                continue
            age = kf.id - mp.first_kf_id
            if mp.n_found / mp.n_visible < 0.25 or (age >= 2 and mp.n_obs() <= 3):
                mp.set_bad()
            elif age < 3:
                keep.append(mp)
        self.recent = keep

    def _triangulate(self, kf: KeyFrame) -> None:
        for n, nb in enumerate(kf.best_covisibles(10)):
            if n > 0 and self.queue:
                return
            if np.linalg.norm(kf.center - nb.center) < kf.camera.baseline:
                continue
            i1, i2 = _epipolar_matches(kf, nb)
            if len(i1):
                self._create_points(kf, nb, i1, i2)

    def _create_points(self, k1: KeyFrame, k2: KeyFrame, i1: np.ndarray, i2: np.ndarray) -> None:
        """ORB-SLAM2 CreateNewMapPoints checks: parallax vs stereo, depth, chi2, scale ratio."""
        cam = k1.camera
        c, f = np.array([cam.cx, cam.cy]), np.array([cam.fx, cam.fy])
        xn1, xn2 = (k1.keypoints[i1] - c) / f, (k2.keypoints[i2] - c) / f
        ray1 = np.c_[xn1, np.ones(len(i1))] @ k1.T_wc[:3, :3].T
        ray2 = np.c_[xn2, np.ones(len(i2))] @ k2.T_wc[:3, :3].T
        cos_rays = np.einsum("ij,ij->i", ray1, ray2) / (np.linalg.norm(ray1, axis=1) * np.linalg.norm(ray2, axis=1))
        d1, d2 = stereo_depth(k1)[i1], stereo_depth(k2)[i2]
        st1, st2 = d1 > 0, d2 > 0
        cps1 = np.where(st1, np.cos(2 * np.arctan2(cam.baseline / 2, np.nan_to_num(d1, nan=1.0))), cos_rays + 1)
        cps2 = np.where(st2, np.cos(2 * np.arctan2(cam.baseline / 2, np.nan_to_num(d2, nan=1.0))), cos_rays + 1)

        dlt = (cos_rays < np.minimum(cps1, cps2)) & (cos_rays > 0) & (st1 | st2 | (cos_rays < 0.9998))
        s1 = ~dlt & st1 & (cps1 < cps2)
        s2 = ~dlt & ~s1 & st2 & (cps2 < cps1)
        X = np.full((len(i1), 3), np.nan)
        if dlt.any():
            X[dlt] = triangulate_dlt(k1.T_cw, k2.T_cw, xn1[dlt], xn2[dlt])
        if s1.any():
            X[s1] = unproject(k1, i1[s1], k1.T_wc)
        if s2.any():
            X[s2] = unproject(k2, i2[s2], k2.T_wc)

        ok = np.isfinite(X).all(axis=1)
        X = np.nan_to_num(X)
        ok &= _reprojection_ok(k1, i1, X) & _reprojection_ok(k2, i2, X)
        dist1 = np.linalg.norm(X - k1.center, axis=1)
        dist2 = np.linalg.norm(X - k2.center, axis=1)
        ratio_dist = dist2 / np.maximum(dist1, 1e-12)
        ratio_oct = SCALES[k1.octave[i1]] / SCALES[k2.octave[i2]]
        factor = 1.5 * SCALE
        ok &= (dist1 > 0) & (dist2 > 0) & (ratio_dist * factor >= ratio_oct) & (ratio_dist <= ratio_oct * factor)

        for a in np.flatnonzero(ok):
            mp = self.map.new_point(X[a], k1, i1[a])
            mp.add_observation(k1, i1[a])
            mp.add_observation(k2, i2[a])
            mp.update_descriptor()
            mp.update_normal_and_depth()
            self.recent.append(mp)

    def _fuse_neighbors(self, kf: KeyFrame) -> None:
        kf.fuse_target = kf.id
        targets = []
        for n in kf.best_covisibles(10):
            for k in [n] + n.best_covisibles(5):
                if not k.bad and k.fuse_target != kf.id:
                    k.fuse_target = kf.id
                    targets.append(k)
        mine = [mp for _, mp in kf.valid_points()]
        for t in targets:
            fuse(t, mine)
        theirs = {mp: None for t in targets for _, mp in t.valid_points()}
        fuse(kf, list(theirs))
        for _, mp in kf.valid_points():
            mp.update_descriptor()
            mp.update_normal_and_depth()
        kf.update_connections()

    def _local_ba(self, kf: KeyFrame) -> None:
        with self.map.lock:
            version = self.map.version
            local = [kf] + kf.best_covisibles()
            for k in local:
                k.ba_local = kf.id
            pts = list({mp: None for k in local for _, mp in k.valid_points()})
            fixed = []
            for mp in pts:
                for k in mp.obs:
                    if k.ba_local != kf.id and k.ba_fixed != kf.id:
                        k.ba_fixed = kf.id
                        fixed.append(k)
            kfs = local + fixed
            kidx = {k: n for n, k in enumerate(kfs)}
            links = [(kidx[k], q, k, i) for q, mp in enumerate(pts) for k, i in mp.obs.items() if k in kidx]
            if len(links) < 10:
                return
            pose_i, pt_i, obs_kf, kp_i = zip(*links)
            z = np.array([[*k.keypoints[i], k.u_right[i]] for k, i in zip(obs_kf, kp_i)])
            info = np.array([INV_SIGMA2[k.octave[i]] for k, i in zip(obs_kf, kp_i)])
            obs = Obs(np.array(pose_i), np.array(pt_i), z, info)
            is_fixed = np.array([n >= len(local) or k is self.map.origin for n, k in enumerate(kfs)])
            poses = [k.T_cw for k in kfs]
            X = np.stack([mp.Xw for mp in pts])

        new_poses, new_X, inl = bundle_adjust(poses, is_fixed, X, obs, kf.camera, stop=lambda: self.abort)

        with self.map.lock:
            if self.map.version != version:
                return  # a loop correction landed while we optimized; these poses are stale
            for a in np.flatnonzero(~inl):
                pts[pt_i[a]].erase_observation(obs_kf[a])
            for k, T, fx in zip(kfs, new_poses, is_fixed):
                if not fx and not k.bad:
                    k.T_cw = T
            for mp, x in zip(pts, new_X):
                if not mp.bad:
                    mp.Xw = x
            update_normals_and_depths(pts)

    def _cull_keyframes(self, kf: KeyFrame) -> None:
        """Drop keyframes whose close points are 90% seen by >= 3 others at the same or finer scale."""
        for k in kf.best_covisibles():
            if k is self.map.origin:
                continue
            depth = stereo_depth(k)
            th = self.th_depth * k.camera.baseline
            n_pts = n_red = 0
            for i, mp in k.valid_points():
                if not depth[i] > 0 or depth[i] > th:
                    continue
                n_pts += 1
                if mp.n_obs() > 3:
                    level = k.octave[i]
                    n = sum(1 for k2, j in mp.obs.items() if k2 is not k and k2.octave[j] <= level + 1)
                    n_red += n >= 3
            if n_pts and n_red > 0.9 * n_pts:
                k.set_bad()


def _reprojection_ok(kf: KeyFrame, idx: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Chi2 test of X against keypoint idx of kf (3 dof with a stereo match, else 2)."""
    Xc = transform_points(kf.T_cw, X)
    u, v, ur = kf.camera.project_stereo(Xc).T
    kp = kf.keypoints[idx]
    kur = kf.u_right[idx]
    st = ~np.isnan(kur)
    e2 = (u - kp[:, 0]) ** 2 + (v - kp[:, 1]) ** 2 + np.where(st, (ur - np.nan_to_num(kur)) ** 2, 0.0)
    th = np.where(st, CHI2_STEREO, CHI2_MONO)
    return (Xc[:, 2] > 0) & (e2 * INV_SIGMA2[kf.octave[idx]] < th)


def _fundamental(k1: KeyFrame, k2: KeyFrame) -> np.ndarray:
    """F12 with x1^T F12 x2 = 0."""
    R12 = k1.T_cw[:3, :3] @ k2.T_cw[:3, :3].T
    t12 = -R12 @ k2.T_cw[:3, 3] + k1.T_cw[:3, 3]
    Kinv = np.linalg.inv(k1.camera.K)
    return Kinv.T @ hat(t12) @ R12 @ Kinv


def _epipolar_matches(k1: KeyFrame, k2: KeyFrame) -> tuple[np.ndarray, np.ndarray]:
    """Unmatched keypoint pairs in the same vocabulary node that satisfy the epipolar constraint."""
    f1 = np.flatnonzero(np.equal(k1.points, None))
    f2 = np.flatnonzero(np.equal(k2.points, None))
    if k1.node is None or k2.node is None or not len(f1) or not len(f2):
        return np.zeros(0, int), np.zeros(0, int)
    a, b = same_group_pairs(k1.node[f1], k2.node[f2])
    p, q = f1[a], f2[b]
    d = hamming(k1.descriptors[p], k2.descriptors[q])

    x1, x2 = k1.keypoints[p], k2.keypoints[q]
    line = np.c_[x1, np.ones(len(p))] @ _fundamental(k1, k2)
    num = line[:, 0] * x2[:, 0] + line[:, 1] * x2[:, 1] + line[:, 2]
    s2 = SCALES[k2.octave[q]] ** 2
    epi = num**2 / (line[:, 0] ** 2 + line[:, 1] ** 2) < 3.84 * s2
    e = k2.camera.K @ (k2.T_cw[:3, :3] @ k1.center + k2.T_cw[:3, 3])
    far = ((x2 - e[:2] / e[2]) ** 2).sum(axis=1) >= 100 * s2
    stereo = ~np.isnan(k1.u_right[p]) | ~np.isnan(k2.u_right[q])
    ok = (d <= TH_LOW) & epi & (stereo | far)
    p, q, _ = one_to_one(p[ok], q[ok], d[ok], TH_LOW)
    keep = rotation_consistent(k1.angle[p], k2.angle[q])
    return p[keep], q[keep]


def fuse(kf: KeyFrame, mps: list[MapPoint], th: float = 3.0) -> None:
    """Project points into `kf`; merge with the matched keypoint's point or add an observation."""
    mps = [mp for mp in mps if not mp.bad and kf not in mp.obs]
    if not mps:
        return
    Xw, normal, dmin, dmax, desc = point_arrays(mps)
    pidx, uv, _, level = visible(Xw, normal, dmin, dmax, kf.T_cw, kf.camera)
    if not len(pidx):
        return

    a, j = window_pairs(kf.keypoints, uv[:, 0], uv[:, 1], th * SCALES[level])
    p = pidx[a]
    oct_j = kf.octave[j]
    ok = (oct_j >= level[a] - 1) & (oct_j <= level[a]) & _reprojection_ok(kf, j, Xw[p])
    p, j = p[ok], j[ok]
    g, k, best, _ = best_two(p, hamming(desc[p], kf.descriptors[j]))
    good = best <= TH_LOW
    for pi, ji in zip(g[good], j[k[good]]):
        mp = mps[pi]
        if mp.bad or kf in mp.obs:
            continue
        other = kf.points[ji]
        if other is None or other.bad:
            mp.add_observation(kf, ji)
        elif other is not mp:
            if other.n_obs() > mp.n_obs():
                mp.replace(other)
            else:
                other.replace(mp)
