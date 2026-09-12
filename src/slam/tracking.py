"""Tracking: per-frame camera pose against the local map (ORB-SLAM2, stereo)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam.data.frame import Frame
from slam.features.matcher import TH_HIGH, project_search, rotation_consistent
from slam.features.orb import INV_SIGMA2, N_LEVELS, SCALES
from slam.geometry.lie import exp_se3, invert, log_se3
from slam.map.structure import KeyFrame, Map, MapPoint, point_arrays, visible
from slam.optimize.solver import motion_only_ba


@dataclass
class TrackCfg:
    th_depth: float = 35.0  # close/far split, in baselines (ORB-SLAM2 KITTI setting)
    max_frames: int = 10  # camera fps
    min_init_points: int = 500


def stereo_depth(frame) -> np.ndarray:
    """Depth per keypoint from its stereo match, NaN for monocular keypoints."""
    cam = frame.camera
    return cam.fx * cam.baseline / (frame.keypoints[:, 0] - frame.u_right)


def _through_ref(ref: KeyFrame, T_cr: np.ndarray) -> np.ndarray:
    """Frame pose from its reference keyframe, walking up the spanning tree past culled ones."""
    while ref.bad:
        T_cr = T_cr @ ref.T_cp
        ref = ref.parent
    return T_cr @ ref.T_cw


def unproject(frame, idx: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    kp = frame.keypoints[idx]
    Xc = frame.camera.triangulate_stereo(kp[:, 0], kp[:, 1], frame.u_right[idx])
    return Xc @ T_wc[:3, :3].T + T_wc[:3, 3]


class Tracker:
    def __init__(self, map_: Map, mapper, cfg: TrackCfg | None = None):
        self.map = map_
        self.mapper = mapper
        self.cfg = cfg or TrackCfg()
        self.state = "init"
        self.last: Frame | None = None
        self.velocity: np.ndarray | None = None
        self.ref_kf: KeyFrame | None = None
        self.last_kf: KeyFrame | None = None
        self.last_reloc = 0
        self.local_kfs: list[KeyFrame] = []
        self.n_inliers = 0
        self.traj: list[tuple[KeyFrame, np.ndarray]] = []

    def track(self, frame: Frame) -> np.ndarray:
        with self.map.lock:
            if self.state == "init":
                self._initialize(frame)
            else:
                self._track(frame)
            if self.ref_kf is not None:
                self.traj.append((self.ref_kf, frame.T_cw @ self.ref_kf.T_wc))
            self.last = frame
        return frame.T_cw

    def trajectory(self) -> list[np.ndarray]:
        """Per-frame T_cw re-expressed through the (optimized) reference keyframes."""
        return [_through_ref(ref, T_cr) for ref, T_cr in self.traj]

    def _initialize(self, frame: Frame) -> None:
        stereo = np.flatnonzero(stereo_depth(frame) > 0)
        if len(stereo) < self.cfg.min_init_points:
            return
        frame.T_cw = np.eye(4)
        kf = self.map.new_keyframe(frame)
        for i, X in zip(stereo, unproject(frame, stereo, np.eye(4))):
            frame.points[i] = self._new_point(X, kf, i)
        self.mapper.insert(kf)
        self.ref_kf = self.last_kf = kf
        self.local_kfs = [kf]
        self.state = "ok"

    def _new_point(self, X: np.ndarray, kf: KeyFrame, i: int) -> MapPoint:
        mp = self.map.new_point(X, kf, i)
        mp.add_observation(kf, i)
        mp.update_normal_and_depth()
        return mp

    def _track(self, frame: Frame) -> None:
        self._update_last_frame()
        ok = False
        if self.state == "ok" and self.velocity is not None and frame.frame_id >= self.last_reloc + 2:
            ok = self._track_motion_model(frame)
        if not ok:
            ok = self._track_wide(frame)
        ok = ok and self._track_local_map(frame)
        if not ok:
            self.state = "lost"
            frame.T_cw = self._predict()
            return
        if self.state == "lost":
            self.last_reloc = frame.frame_id
        self.state = "ok"
        # Kept as a twist: exp() of it is always a rotation, so repeated prediction cannot
        # walk the pose off SE(3) (composing T @ invert(T_prev) does, ~3x per frame).
        self.velocity = log_se3(frame.T_cw @ invert(self.last.T_cw))
        if self._need_keyframe(frame):
            self._create_keyframe(frame)
        frame.points[frame.outlier] = None

    def _update_last_frame(self) -> None:
        # Re-anchor on the reference keyframe: BA or a loop closure may have moved it since.
        self.last.T_cw = _through_ref(*self.traj[-1])
        pts = self.last.points
        for i, mp in enumerate(pts):
            while mp is not None and mp.replaced is not None:
                mp = mp.replaced
            pts[i] = None if mp is None or mp.bad else mp

    def _predict(self) -> np.ndarray:
        if self.velocity is None:
            return self.last.T_cw.copy()
        return exp_se3(self.velocity) @ self.last.T_cw

    def _track_motion_model(self, frame: Frame) -> bool:
        frame.T_cw = self._predict()
        n = self._search_last(frame, 7.0)
        if n < 20:
            n = self._search_last(frame, 14.0)
        return n >= 20 and self._optimize_pose(frame) >= 10

    def _track_wide(self, frame: Frame) -> bool:
        """Fallback: wide projection search from the predicted pose (BoW reloc comes later)."""
        frame.T_cw = self._predict()
        return self._search_last(frame, 30.0) >= 15 and self._optimize_pose(frame) >= 10

    def _search_last(self, frame: Frame, th: float) -> int:
        """Project the last frame's map points; octave window follows the motion direction."""
        frame.points[:] = None
        last = self.last
        idx = np.flatnonzero(np.not_equal(last.points, None))
        if len(idx) == 0:
            return 0
        mps = last.points[idx]
        tz = (last.T_cw @ invert(frame.T_cw))[2, 3]
        b = frame.camera.baseline
        o = last.octave[idx]
        lo = np.where(tz > b, o, np.where(-tz > b, 0, o - 1))
        hi = np.where(tz > b, N_LEVELS - 1, np.where(-tz > b, o, o + 1))
        Xw = np.stack([mp.Xw for mp in mps])
        desc = np.stack([mp.descriptor for mp in mps])
        p, j, _ = project_search(frame, Xw, desc, th * SCALES[o], lo, hi, TH_HIGH)
        keep = rotation_consistent(last.angle[idx[p]], frame.angle[j])
        frame.points[j[keep]] = mps[p[keep]]
        return int(keep.sum())

    def _optimize_pose(self, frame: Frame, count_found: bool = False) -> int:
        idx = np.flatnonzero(np.not_equal(frame.points, None))
        frame.outlier[:] = False
        if len(idx) < 3:
            return 0
        X = np.stack([mp.Xw for mp in frame.points[idx]])
        z = np.column_stack([frame.keypoints[idx], frame.u_right[idx]])
        frame.T_cw, inl = motion_only_ba(frame.T_cw, X, z, INV_SIGMA2[frame.octave[idx]], frame.camera)
        frame.outlier[idx[~inl]] = True
        if count_found:
            for mp in frame.points[idx[inl]]:
                mp.n_found += 1
        else:
            frame.points[idx[~inl]] = None
        return int(inl.sum())

    def _track_local_map(self, frame: Frame) -> bool:
        self._update_local_keyframes(frame)
        seen: dict[MapPoint, None] = {}
        for kf in self.local_kfs:
            for _, mp in kf.valid_points():
                seen[mp] = None
        self._search_local_points(frame, list(seen))
        self.n_inliers = self._optimize_pose(frame, count_found=True)
        recent = frame.frame_id < self.last_reloc + self.cfg.max_frames
        return self.n_inliers >= (50 if recent else 30)

    def _update_local_keyframes(self, frame: Frame) -> None:
        """K1 = keyframes sharing points with the frame; K2 = one covisible, child, parent each."""
        counts: dict[KeyFrame, int] = {}
        for i, mp in enumerate(frame.points):
            if mp is None:
                continue
            if mp.bad:
                frame.points[i] = None
                continue
            for kf in mp.obs:
                counts[kf] = counts.get(kf, 0) + 1
        k1 = [kf for kf in sorted(counts, key=counts.get, reverse=True) if not kf.bad]
        if not k1:
            return
        fid = frame.frame_id
        for kf in k1:
            kf.track_ref = fid
        local = list(k1)
        for kf in k1:
            if len(local) > 80:
                break
            for group in (kf.best_covisibles(10), kf.children, [kf.parent] if kf.parent else []):
                for n in group:
                    if not n.bad and n.track_ref != fid:
                        n.track_ref = fid
                        local.append(n)
                        break
        self.local_kfs = local
        self.ref_kf = k1[0]

    def _search_local_points(self, frame: Frame, local: list[MapPoint]) -> None:
        fid = frame.frame_id
        for mp in frame.points:
            if mp is not None:
                mp.n_visible += 1
                mp.last_frame = fid
        cand = [mp for mp in local if mp.last_frame != fid and not mp.bad]
        if not cand:
            return
        Xw, normal, dmin, dmax, desc = point_arrays(cand)
        idx, _, view, level = visible(Xw, normal, dmin, dmax, frame.T_cw, frame.camera)
        for k in idx:
            cand[k].n_visible += 1
        th = 3.0 if fid < self.last_reloc + 2 else 1.0
        radius = th * SCALES[level] * np.where(view > 0.998, 2.5, 4.0)
        p, j, _ = project_search(frame, Xw[idx], desc[idx], radius, level - 1, level, TH_HIGH, ratio=0.8)
        for a, b in zip(p, j):
            frame.points[b] = cand[idx[a]]

    def _need_keyframe(self, frame: Frame) -> bool:
        n_kfs = len(self.map.keyframes)
        if frame.frame_id < self.last_reloc + self.cfg.max_frames and n_kfs > self.cfg.max_frames:
            return False
        ref_matches = self.ref_kf.tracked_points(3 if n_kfs > 2 else 2)
        idle = self.mapper.idle()

        depth = stereo_depth(frame)
        close = depth < self.cfg.th_depth * frame.camera.baseline
        tracked = np.not_equal(frame.points, None) & ~frame.outlier
        need_close = (close & tracked).sum() < 100 and (close & ~tracked).sum() > 70

        since = frame.frame_id - self.last_kf.frame_id
        c1 = since >= self.cfg.max_frames or idle
        c1 = c1 or self.n_inliers < 0.25 * ref_matches or need_close
        th_ref = 0.75 if n_kfs >= 2 else 0.4
        c2 = (self.n_inliers < th_ref * ref_matches or need_close) and self.n_inliers > 15
        if not (c1 and c2):
            return False
        if idle:
            return True
        self.mapper.interrupt_ba()
        return self.mapper.queue_len() < 3

    def _create_keyframe(self, frame: Frame) -> None:
        """New keyframe plus map points from its closest stereo keypoints (>= 100, all close ones)."""
        kf = self.map.new_keyframe(frame)
        depth = stereo_depth(frame)
        cand = np.flatnonzero(depth > 0)
        cand = cand[np.argsort(depth[cand])]
        new = [i for i in cand if frame.points[i] is None]
        X = dict(zip(new, unproject(frame, np.array(new, dtype=int), kf.T_wc))) if new else {}
        th = self.cfg.th_depth * frame.camera.baseline
        for n, i in enumerate(cand, 1):
            if frame.points[i] is None:
                frame.points[i] = self._new_point(X[i], kf, i)
            if depth[i] > th and n > 100:
                break
        self.ref_kf = self.last_kf = kf
        self.mapper.insert(kf)
