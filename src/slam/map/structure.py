"""Map points, keyframes, covisibility graph, spanning tree, and the map (ORB-SLAM2 semantics)."""

from __future__ import annotations

import threading

import numpy as np

from slam.data.frame import Frame
from slam.features.matcher import hamming
from slam.features.orb import SCALES, predict_level
from slam.geometry.lie import invert, transform_points

MIN_COVIS = 15


class MapPoint:
    def __init__(self, map_: Map, Xw: np.ndarray, kf: KeyFrame, idx: int):
        self.map = map_
        self.Xw = np.asarray(Xw, dtype=np.float64)
        self.ref_kf = kf
        self.first_kf_id = kf.id
        self.obs: dict[KeyFrame, int] = {}
        self.descriptor = kf.descriptors[idx]
        self.normal = np.zeros(3)
        self.dmin = self.dmax = 0.0
        self.n_visible = self.n_found = 1
        self.bad = False
        self.replaced: MapPoint | None = None
        self.color = float(kf.gray[idx])
        self.last_frame = -1  # last frame that projected this point in a search

    def n_obs(self) -> int:
        """Stereo observations count twice, as in ORB-SLAM2."""
        return sum(1 if np.isnan(kf.u_right[i]) else 2 for kf, i in self.obs.items())

    def add_observation(self, kf: KeyFrame, idx: int) -> None:
        self.obs[kf] = idx
        kf.points[idx] = self

    def erase_observation(self, kf: KeyFrame) -> None:
        idx = self.obs.pop(kf, None)
        if idx is None:
            return
        if kf.points[idx] is self:
            kf.points[idx] = None
        if self.ref_kf is kf and self.obs:
            self.ref_kf = next(iter(self.obs))
        if self.n_obs() <= 2:
            self.set_bad()

    def set_bad(self) -> None:
        self.bad = True
        for kf, i in self.obs.items():
            if kf.points[i] is self:
                kf.points[i] = None
        self.obs = {}
        self.map.points.discard(self)

    def replace(self, other: MapPoint) -> None:
        if other is self:
            return
        for kf, i in list(self.obs.items()):
            if kf in other.obs:
                kf.points[i] = None
            else:
                other.add_observation(kf, i)
        other.n_found += self.n_found
        other.n_visible += self.n_visible
        other.update_descriptor()
        self.obs = {}
        self.bad = True
        self.replaced = other
        self.map.points.discard(self)

    def update_descriptor(self) -> None:
        """Medoid of all observed descriptors (least median Hamming distance)."""
        if not self.obs:
            return
        D = np.stack([kf.descriptors[i] for kf, i in self.obs.items()])
        self.descriptor = D[np.argmin(np.median(hamming(D[:, None], D[None]), axis=1))]

    def update_normal_and_depth(self) -> None:
        if not self.obs or self.ref_kf not in self.obs:
            return
        v = self.Xw - np.stack([kf.center for kf in self.obs])
        n = (v / np.linalg.norm(v, axis=1, keepdims=True)).mean(axis=0)
        self.normal = n / np.linalg.norm(n)
        ref = self.ref_kf
        self.dmax = np.linalg.norm(self.Xw - ref.center) * SCALES[ref.octave[self.obs[ref]]]
        self.dmin = self.dmax / SCALES[-1]


def visible(Xw, normal, dmin, dmax, T_cw, cam):
    """ORB-SLAM's frustum test: in front, inside the image, within the scale-invariance
    distance range, and seen from under 60 degrees. Returns the visible points'
    (indices, pixels, viewing cosine, predicted pyramid level)."""
    Xc = transform_points(T_cw, Xw)
    PO = Xw - invert(T_cw)[:3, 3]
    dist = np.linalg.norm(PO, axis=1)
    view = np.einsum("ij,ij->i", PO, normal) / dist
    uv = cam.project(Xc)
    ok = (Xc[:, 2] > 0) & cam.in_image(uv) & (dist >= 0.8 * dmin) & (dist <= 1.2 * dmax) & (view >= 0.5)
    idx = np.flatnonzero(ok)
    return idx, uv[idx], view[idx], predict_level(dmax[idx], dist[idx])


def point_arrays(mps: list[MapPoint]):
    """Stacked (Xw, normal, dmin, dmax, descriptor) for vectorized searches."""
    return (
        np.stack([mp.Xw for mp in mps]),
        np.stack([mp.normal for mp in mps]),
        np.array([mp.dmin for mp in mps]),
        np.array([mp.dmax for mp in mps]),
        np.stack([mp.descriptor for mp in mps]),
    )


def update_normals_and_depths(mps: list[MapPoint]) -> None:
    """Batched MapPoint.update_normal_and_depth (used after BA moves thousands of points)."""
    mps = [mp for mp in mps if not mp.bad and mp.obs and mp.ref_kf in mp.obs]
    if not mps:
        return
    pid = np.array([n for n, mp in enumerate(mps) for _ in mp.obs])
    centers = np.array([kf.center for mp in mps for kf in mp.obs])
    X = np.stack([mp.Xw for mp in mps])
    v = X[pid] - centers
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    n = np.zeros_like(X)
    np.add.at(n, pid, v)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    ref = np.array([mp.ref_kf.center for mp in mps])
    level = np.array([mp.ref_kf.octave[mp.obs[mp.ref_kf]] for mp in mps])
    dmax = np.linalg.norm(X - ref, axis=1) * SCALES[level]
    for mp, ni, d in zip(mps, n, dmax):
        mp.normal, mp.dmax, mp.dmin = ni, d, d / SCALES[-1]


class KeyFrame:
    def __init__(self, map_: Map, kid: int, frame: Frame):
        self.map = map_
        self.id = kid
        self.frame_id = frame.frame_id
        self.camera = frame.camera
        self.keypoints = frame.keypoints
        self.descriptors = frame.descriptors
        self.octave = frame.octave
        self.angle = frame.angle
        self.u_right = frame.u_right
        self.points = frame.points.copy()
        uv = np.round(frame.keypoints).astype(int)
        self.gray = frame.image_left[uv[:, 1], uv[:, 0]] / 255.0  # point colors for the viewer
        self.T_cw = frame.T_cw
        self.covis: dict[KeyFrame, int] = {}
        self.ordered: list[KeyFrame] = []
        self.parent: KeyFrame | None = None
        self.children: set[KeyFrame] = set()
        self.loop_edges: set[KeyFrame] = set()
        self.bad = False
        self.T_cp: np.ndarray | None = None  # pose relative to parent, kept once culled
        # Set by Vocabulary.compute: vocabulary node per keypoint, (ids, weights) BoW vector
        self.node: np.ndarray | None = None
        self.bow: tuple[np.ndarray, np.ndarray] | None = None
        # Per-operation stamps so a keyframe is visited once (ORB-SLAM's mnBALocalForKF etc.)
        self.track_ref = self.ba_local = self.ba_fixed = self.fuse_target = -1

    @property
    def T_cw(self) -> np.ndarray:
        return self._T_cw

    @T_cw.setter
    def T_cw(self, T: np.ndarray) -> None:
        self._T_cw = np.array(T, dtype=np.float64)
        self.T_wc = invert(self._T_cw)
        self.center = self.T_wc[:3, 3]

    def valid_points(self) -> list[tuple[int, MapPoint]]:
        return [(i, mp) for i, mp in enumerate(self.points) if mp is not None and not mp.bad]

    def tracked_points(self, min_obs: int = 0) -> int:
        return sum(1 for _, mp in self.valid_points() if min_obs == 0 or mp.n_obs() >= min_obs)

    def update_connections(self) -> None:
        counts: dict[KeyFrame, int] = {}
        for _, mp in self.valid_points():
            for kf in mp.obs:
                if kf is not self and not kf.bad:
                    counts[kf] = counts.get(kf, 0) + 1
        if not counts:
            return
        strong = {kf: w for kf, w in counts.items() if w >= MIN_COVIS}
        if not strong:
            kf = max(counts, key=counts.get)
            strong = {kf: counts[kf]}
        for kf, w in strong.items():
            kf._connect(self, w)
        self.covis = strong
        self._reorder()
        if self.parent is None and self is not self.map.origin:
            self.change_parent(self.ordered[0])

    def _connect(self, kf: KeyFrame, w: int) -> None:
        self.covis[kf] = w
        self._reorder()

    def _disconnect(self, kf: KeyFrame) -> None:
        if self.covis.pop(kf, None) is not None:
            self._reorder()

    def _reorder(self) -> None:
        self.ordered = sorted(self.covis, key=self.covis.get, reverse=True)

    def best_covisibles(self, n: int | None = None) -> list[KeyFrame]:
        good = [k for k in self.ordered if not k.bad]
        return good if n is None else good[:n]

    def change_parent(self, p: KeyFrame) -> None:
        if self.parent is not None:
            self.parent.children.discard(self)
        self.parent = p
        p.children.add(self)

    def set_bad(self) -> bool:
        """Cull this keyframe, re-attaching its spanning-tree children (ORB-SLAM2)."""
        if self is self.map.origin or self.parent is None:
            return False
        for kf in list(self.covis):
            kf._disconnect(self)
        for mp in list(self.points):
            if mp is not None:
                mp.erase_observation(self)
        self.covis, self.ordered = {}, []

        cands = {self.parent}
        kids = {c for c in self.children if not c.bad}
        while kids:
            best, bw = None, -1
            for c in kids:
                for k, w in c.covis.items():
                    if k in cands and w > bw:
                        best, bw = (c, k), w
            if best is None:
                break
            c, p = best
            c.change_parent(p)
            cands.add(c)
            kids.discard(c)
        for c in kids:
            c.change_parent(self.parent)
        self.parent.children.discard(self)
        self.T_cp = self.T_cw @ self.parent.T_wc
        self.bad = True
        self.map.keyframes.pop(self.id, None)
        return True


class Map:
    def __init__(self) -> None:
        self.keyframes: dict[int, KeyFrame] = {}
        self.points: set[MapPoint] = set()
        self.lock = threading.RLock()
        self.origin: KeyFrame | None = None
        self.version = 0  # bumped by loop correction so in-flight BA results can be discarded
        self._next_kf = 0

    def new_keyframe(self, frame: Frame) -> KeyFrame:
        kf = KeyFrame(self, self._next_kf, frame)
        self._next_kf += 1
        self.keyframes[kf.id] = kf
        if self.origin is None:
            self.origin = kf
        return kf

    def new_point(self, Xw: np.ndarray, kf: KeyFrame, idx: int) -> MapPoint:
        mp = MapPoint(self, Xw, kf, idx)
        self.points.add(mp)
        return mp
