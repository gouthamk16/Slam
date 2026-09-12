"""Vectorized descriptor matching primitives."""

from __future__ import annotations

import numpy as np

from slam.geometry.lie import transform_points

TH_HIGH = 100
TH_LOW = 50
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamming distance over the last axis of broadcastable uint8 descriptor arrays."""
    return _POP[np.bitwise_xor(a, b)].sum(axis=-1, dtype=np.int32)


def ragged_pairs(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All (i, j) with lo[i] <= j < hi[i], as flat index arrays."""
    cnt = np.maximum(hi - lo, 0)
    i = np.repeat(np.arange(len(lo)), cnt)
    j = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
    return i, j


def same_group_pairs(g1: np.ndarray, g2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All (i, j) with g1[i] == g2[j] (e.g. keypoints in the same vocabulary node)."""
    order = np.argsort(g2)
    gs = g2[order]
    i, b = ragged_pairs(np.searchsorted(gs, g1), np.searchsorted(gs, g1, "right"))
    return i, order[b]


def best_two(group: np.ndarray, dist: np.ndarray):
    """For each distinct group: (group, index of its best pair, best dist, second-best dist)."""
    if len(group) == 0:
        e = np.zeros(0, dtype=int)
        return e, e, e, e
    order = np.lexsort((dist, group))
    g, d = group[order], dist[order]
    first = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    nxt = np.minimum(first + 1, len(g) - 1)
    has2 = (first + 1 < len(g)) & (g[nxt] == g[first])
    second = np.where(has2, d[nxt], np.iinfo(np.int32).max)
    return g[first], order[first], d[first], second


def one_to_one(p: np.ndarray, q: np.ndarray, d: np.ndarray, max_dist: int, ratio: float | None = None):
    """Mutually closest (p, q) pairs: best q per p (optional ratio test), then best p per q."""
    g, k, best, second = best_two(p, d)
    keep = best <= max_dist
    if ratio is not None:
        keep &= best < ratio * second
    p, q, d = g[keep], q[k[keep]], best[keep]
    qj, k2, _, _ = best_two(q, d)
    return p[k2], qj, d[k2]


def window_pairs(kp: np.ndarray, u: np.ndarray, v: np.ndarray, radius: np.ndarray):
    """(point, keypoint) index pairs with the keypoint inside the point's square window."""
    order = np.argsort(kp[:, 0])
    ku = kp[order, 0]
    a, b = ragged_pairs(np.searchsorted(ku, u - radius), np.searchsorted(ku, u + radius, "right"))
    j = order[b]
    ok = np.abs(kp[j, 1] - v[a]) <= radius[a]
    return a[ok], j[ok]


def project_search(frame, Xw, desc, radius, oct_lo, oct_hi, max_dist, ratio=None, T_cw=None, free=None):
    """Match world points to free keypoints of `frame` around their projections.

    Candidates lie within `radius` (per point) in u, v and, for stereo keypoints, u_right,
    with octave in [oct_lo, oct_hi]. `T_cw` overrides the frame pose and `free` the mask of
    assignable keypoints (default: keypoints without a map point). Returns
    (point idx, keypoint idx, distance), one-to-one, keeping the closest descriptor on conflicts.
    """
    cam = frame.camera
    T = frame.T_cw if T_cw is None else T_cw
    free = np.equal(frame.points, None) if free is None else free
    Xc = transform_points(T, Xw)
    uvr = cam.project_stereo(Xc)
    pidx = np.flatnonzero((Xc[:, 2] > 0) & cam.in_image(uvr[:, :2]))

    a, j = window_pairs(frame.keypoints, uvr[pidx, 0], uvr[pidx, 1], radius[pidx])
    p = pidx[a]
    kur = frame.u_right[j]
    ok = (
        (frame.octave[j] >= oct_lo[p])
        & (frame.octave[j] <= oct_hi[p])
        & free[j]
        & (np.isnan(kur) | (np.abs(np.nan_to_num(kur) - uvr[p, 2]) <= radius[p]))
    )
    p, j = p[ok], j[ok]
    return one_to_one(p, j, hamming(desc[p], frame.descriptors[j]), max_dist, ratio)


def rotation_consistent(a_src: np.ndarray, a_dst: np.ndarray, bins: int = 30) -> np.ndarray:
    """ORB-SLAM orientation check: keep matches in the three dominant rotation bins."""
    b = np.round(((a_src - a_dst) % 360.0) * bins / 360.0).astype(int) % bins
    cnt = np.bincount(b, minlength=bins)
    top = np.argsort(-cnt)[:3]
    top = top[cnt[top] >= 0.1 * cnt[top[0]]]
    return np.isin(b, top)
