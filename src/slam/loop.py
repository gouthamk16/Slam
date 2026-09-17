"""Loop closing (ORB-SLAM2, stereo): BoW candidates with covisibility consistency, SE(3)
RANSAC verification, loop fusion, and essential-graph pose-graph optimization.

Stereo makes scale observable, so the Sim(3) steps of monocular ORB-SLAM become SE(3).
"""

from __future__ import annotations

import numpy as np

from slam.features.matcher import TH_LOW, hamming, one_to_one, project_search, rotation_consistent, same_group_pairs
from slam.features.orb import INV_SIGMA2, SCALES
from slam.features.vocab import score
from slam.geometry.align import horn_se3
from slam.geometry.lie import invert
from slam.map.structure import KeyFrame, Map, MapPoint, point_arrays, visible
from slam.mapping import fuse
from slam.optimize.pose_graph import pose_graph_se3
from slam.optimize.solver import motion_only_ba
from slam.worker import Worker

CONSISTENCY = 3
MIN_COVIS_EDGE = 100


class KeyFrameDB:
    """Inverted file: vocabulary word -> keyframes containing it."""

    def __init__(self) -> None:
        self.inv: dict[int, list[KeyFrame]] = {}

    def add(self, kf: KeyFrame) -> None:
        for w in kf.bow[0]:
            self.inv.setdefault(int(w), []).append(kf)

    def candidates(self, kf: KeyFrame, min_score: float, exclude: set[KeyFrame]) -> list[KeyFrame]:
        common: dict[KeyFrame, int] = {}
        for w in kf.bow[0]:
            for k in self.inv.get(int(w), ()):
                if not k.bad and k not in exclude:
                    common[k] = common.get(k, 0) + 1
        if not common:
            return []
        th = 0.8 * max(common.values())
        scores = {k: s for k, n in common.items() if n > th and (s := score(kf.bow, k.bow)) >= min_score}
        # Accumulate over covisible neighbours so a place seen by a cluster of KFs wins.
        acc: dict[KeyFrame, float] = {}
        for k, s in scores.items():
            best, total = k, s
            for n in k.best_covisibles(10):
                if n in scores:
                    total += scores[n]
                    if scores[n] > scores[best]:
                        best = n
            acc[best] = max(acc.get(best, 0.0), total)
        if not acc:
            return []
        top = max(acc.values())
        return [k for k, t in acc.items() if t > 0.75 * top]


class LoopCloser(Worker):
    def __init__(self, map_: Map, threaded: bool = False):
        super().__init__(threaded)
        self.map = map_
        self.db = KeyFrameDB()
        self.groups: list[tuple[set[KeyFrame], int]] = []
        self.last_loop = 0
        self.loops: list[tuple[int, int]] = []  # (current frame id, matched frame id)

    def process(self, kf: KeyFrame) -> None:
        if kf.bow is None:
            return
        if not kf.bad:
            with self.map.lock:
                cands = self._detect(kf)
                found = self._verify(kf, cands) if cands else None
                if found is not None:
                    self._correct(kf, *found)
        self.db.add(kf)

    def _detect(self, kf: KeyFrame) -> list[KeyFrame]:
        """Candidates that stayed consistent (covisible groups overlap) for 3 consecutive keyframes."""
        if kf.id < self.last_loop + 10:
            return []
        cov = kf.best_covisibles()
        min_score = min((score(kf.bow, c.bow) for c in cov), default=0.0)
        cands = self.db.candidates(kf, min_score, set(cov) | {kf})
        groups, enough, extended = [], [], set()
        for c in cands:
            group = set(c.best_covisibles()) | {c}
            consistent = False
            for g, (prev, count) in enumerate(self.groups):
                if group & prev:
                    consistent = True
                    # Each previous group is extended once, otherwise groups multiply every
                    # keyframe when several candidates overlap (hovering in one room does this).
                    if g not in extended:
                        extended.add(g)
                        groups.append((group, count + 1))
                    if count + 1 >= CONSISTENCY and c not in enough:
                        enough.append(c)
            if not consistent:
                groups.append((group, 0))
        self.groups = groups
        return enough

    def _verify(self, kf: KeyFrame, cands: list[KeyFrame]):
        for c in cands:
            i1, i2 = _bow_matches(kf, c)
            if len(i1) < 20:
                continue
            ransac = _ransac_se3(kf, c, i1, i2)
            if ransac is None:
                continue
            T12, inl = ransac
            T = T12 @ c.T_cw
            matched = np.full(len(kf.keypoints), None, dtype=object)
            matched[i1[inl]] = c.points[i2[inl]]
            loop_pts = list({mp: None for k in [c] + c.best_covisibles() for _, mp in k.valid_points()})
            _search_loop(kf, T, loop_pts, matched)
            idx = np.flatnonzero(np.not_equal(matched, None))
            X = np.stack([mp.Xw for mp in matched[idx]])
            z = np.column_stack([kf.keypoints[idx], kf.u_right[idx]])
            T, good = motion_only_ba(T, X, z, INV_SIGMA2[kf.octave[idx]], kf.camera)
            if good.sum() >= 40:
                matched[idx[~good]] = None
                return c, T, matched, loop_pts
        return None

    def _correct(self, kf: KeyFrame, match: KeyFrame, T_kf: np.ndarray, matched: np.ndarray, loop_pts: list[MapPoint]) -> None:
        self.map.version += 1
        kf.update_connections()
        connected = [kf] + kf.best_covisibles()
        old = {k: k.T_cw for k in connected}  # only these move before the pose graph runs
        corrected = {k: (k.T_cw @ kf.T_wc) @ T_kf for k in connected}

        # Move connected keyframes and their points with the loop correction.
        moved: set[MapPoint] = set()
        for k in connected:
            corr = invert(corrected[k]) @ k.T_cw
            for _, mp in k.valid_points():
                if mp not in moved:
                    moved.add(mp)
                    mp.Xw = corr[:3, :3] @ mp.Xw + corr[:3, 3]
            k.T_cw = corrected[k]
        for mp in moved:
            mp.update_normal_and_depth()

        # Fuse duplicated points on both sides of the loop.
        for i in np.flatnonzero(np.not_equal(matched, None)):
            lp, cur = matched[i], kf.points[i]
            if lp.bad:
                continue
            if cur is None or cur.bad:
                lp.add_observation(kf, i)
            elif cur is not lp:
                cur.replace(lp)
        before = {k: set(k.best_covisibles()) for k in connected}
        for k in connected:
            fuse(k, loop_pts, th=4.0)
        new_links = {}
        for k in connected:
            k.update_connections()
            new_links[k] = set(k.best_covisibles()) - before[k] - set(connected)

        self._optimize_essential_graph(match, old, new_links)
        match.loop_edges.add(kf)
        kf.loop_edges.add(match)
        self.last_loop = kf.id
        self.loops.append((kf.frame_id, match.frame_id))

    def _optimize_essential_graph(self, match: KeyFrame, old: dict, new_links: dict) -> None:
        """Spanning tree + strong covisibility + loop edges, measured with pre-correction poses;
        new loop links use corrected poses. The matched keyframe fixes the gauge."""
        kfs = sorted(self.map.keyframes.values(), key=lambda k: k.id)
        idx = {k: n for n, k in enumerate(kfs)}
        edges: dict[tuple[int, int], np.ndarray] = {}

        def add(a: KeyFrame, b: KeyFrame, Ta: np.ndarray, Tb: np.ndarray) -> None:
            if a in idx and b in idx and a is not b:
                key = (idx[a], idx[b]) if idx[a] < idx[b] else (idx[b], idx[a])
                if key not in edges:
                    i, j = key
                    Ti, Tj = (Ta, Tb) if idx[a] == i else (Tb, Ta)
                    edges[key] = Tj @ invert(Ti)

        # Loop links first: after fusion the same pairs are also strongly covisible, and a
        # covisibility edge measured with drifted poses would otherwise encode the drift.
        for k, links in new_links.items():
            for n in links:
                add(k, n, k.T_cw, n.T_cw)
        for k in kfs:
            o = old.get(k, k.T_cw)
            if k.parent is not None and not k.parent.bad:
                add(k.parent, k, old.get(k.parent, k.parent.T_cw), o)
            for n in k.loop_edges:
                add(n, k, old.get(n, n.T_cw), o)
            for n, w in k.covis.items():
                if w >= MIN_COVIS_EDGE:
                    add(n, k, old.get(n, n.T_cw), o)

        before = [k.T_cw for k in kfs]
        after = pose_graph_se3(before, [(i, j, T) for (i, j), T in edges.items()], fixed=(idx[match],))
        corr = {k: invert(Ta) @ Tb for k, Ta, Tb in zip(kfs, after, before)}
        for mp in self.map.points:
            ref = mp.ref_kf if mp.ref_kf in corr else next((k for k in mp.obs if k in corr), None)
            if ref is not None:
                C = corr[ref]
                mp.Xw = C[:3, :3] @ mp.Xw + C[:3, 3]
                mp.update_normal_and_depth()
        for k, T in zip(kfs, after):
            k.T_cw = T
        self.map.version += 1


def _bow_matches(k1: KeyFrame, k2: KeyFrame) -> tuple[np.ndarray, np.ndarray]:
    """Keypoints with map points in both keyframes, matched within vocabulary nodes."""
    f1 = np.flatnonzero([mp is not None and not mp.bad for mp in k1.points])
    f2 = np.flatnonzero([mp is not None and not mp.bad for mp in k2.points])
    if not len(f1) or not len(f2):
        return np.zeros(0, int), np.zeros(0, int)
    a, b = same_group_pairs(k1.node[f1], k2.node[f2])
    p, q = f1[a], f2[b]
    p, q, _ = one_to_one(p, q, hamming(k1.descriptors[p], k2.descriptors[q]), TH_LOW, ratio=0.75)
    keep = rotation_consistent(k1.angle[p], k2.angle[q])
    return p[keep], q[keep]


def _ransac_se3(k1: KeyFrame, k2: KeyFrame, i1: np.ndarray, i2: np.ndarray, iters: int = 300):
    """T_c1c2 from 3D-3D Horn hypotheses; inliers reproject within chi2(2, 99%) in both views."""
    cam = k1.camera
    P1 = np.stack([k1.points[i].Xw for i in i1]) @ k1.T_cw[:3, :3].T + k1.T_cw[:3, 3]
    P2 = np.stack([k2.points[j].Xw for j in i2]) @ k2.T_cw[:3, :3].T + k2.T_cw[:3, 3]
    th1 = 9.21 * SCALES[k1.octave[i1]] ** 2
    th2 = 9.21 * SCALES[k2.octave[i2]] ** 2

    def inliers(T: np.ndarray) -> np.ndarray:
        Q1 = P2 @ T[:3, :3].T + T[:3, 3]
        Ti = invert(T)
        Q2 = P1 @ Ti[:3, :3].T + Ti[:3, 3]
        e1 = ((cam.project(Q1) - k1.keypoints[i1]) ** 2).sum(axis=1)
        e2 = ((cam.project(Q2) - k2.keypoints[i2]) ** 2).sum(axis=1)
        return (Q1[:, 2] > 0) & (Q2[:, 2] > 0) & (e1 < th1) & (e2 < th2)

    rng = np.random.default_rng(0)
    best = np.zeros(len(i1), dtype=bool)
    for _ in range(iters):
        s = rng.choice(len(i1), 3, replace=False)
        inl = inliers(horn_se3(P2[s], P1[s]))
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 20:
        return None
    T = horn_se3(P2[best], P1[best])
    inl = inliers(T)
    return (T, inl) if inl.sum() >= 20 else None


def _search_loop(kf: KeyFrame, T_cw: np.ndarray, mps: list[MapPoint], matched: np.ndarray, th: float = 10.0) -> None:
    """Project loop-side points into `kf` at its corrected pose; fill unmatched keypoints."""
    have = {mp for mp in matched if mp is not None}
    mps = [mp for mp in mps if mp not in have and not mp.bad]
    if not mps:
        return
    Xw, normal, dmin, dmax, desc = point_arrays(mps)
    idx, _, _, level = visible(Xw, normal, dmin, dmax, T_cw, kf.camera)
    p, j, _ = project_search(
        kf, Xw[idx], desc[idx], th * SCALES[level], level - 1, level, TH_LOW,
        T_cw=T_cw, free=np.equal(matched, None),
    )
    for a, b in zip(p, j):
        matched[b] = mps[idx[a]]
