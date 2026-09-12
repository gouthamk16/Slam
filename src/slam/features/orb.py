"""ORB features with pyramid octaves, spatial bucketing, and ORB-SLAM2 stereo matching."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from slam.data.frame import Frame
from slam.features.matcher import TH_HIGH, TH_LOW, best_two, hamming, ragged_pairs

N_LEVELS = 8
SCALE = 1.2
SCALES = SCALE ** np.arange(N_LEVELS)
INV_SIGMA2 = 1.0 / SCALES**2
_W, _L = 5, 5  # SAD half-window and search half-range, in pyramid-level pixels
_extractors: dict[tuple[int, str], cv2.ORB] = {}
_pool = ThreadPoolExecutor(1)  # runs the right-image extraction alongside the left


def predict_level(dmax: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Pyramid level at which a point with scale-invariance distance dmax appears at `dist`."""
    return np.clip(np.ceil(np.log(dmax / np.maximum(dist, 1e-12)) / np.log(SCALE)), 0, N_LEVELS - 1).astype(int)


def extract_orb(image: np.ndarray, n_features: int = 2000, cell: int = 40, tag: str = "L"):
    """Keypoints (N,2), descriptors (N,32), octave (N,), angle (N,).

    Detects 2x the budget, then keeps the strongest per grid cell so features spread
    over the image instead of piling onto trees.
    """
    key = (n_features, tag)  # one extractor per stream so left/right can run concurrently
    if key not in _extractors:
        _extractors[key] = cv2.ORB_create(
            nfeatures=2 * n_features,
            scaleFactor=SCALE,
            nlevels=N_LEVELS,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=12,
        )
    kps, desc = _extractors[key].detectAndCompute(image, None)
    if not kps:
        return np.zeros((0, 2)), np.zeros((0, 32), np.uint8), np.zeros(0, int), np.zeros(0)
    pts = cv2.KeyPoint_convert(kps).astype(np.float64)
    resp = np.array([k.response for k in kps])
    octv = np.array([k.octave for k in kps])
    ang = np.array([k.angle for k in kps])
    keep = _bucket(pts, resp, n_features, cell)
    return pts[keep], desc[keep], octv[keep], ang[keep]


def _bucket(pts: np.ndarray, resp: np.ndarray, n: int, cell: int) -> np.ndarray:
    if len(pts) <= n:
        return np.arange(len(pts))
    cid = (pts[:, 1] // cell).astype(int) * 10000 + (pts[:, 0] // cell).astype(int)
    order = np.lexsort((-resp, cid))
    c = cid[order]
    start = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    rank = np.arange(len(c)) - np.repeat(start, np.diff(np.r_[start, len(c)]))
    cand = order[rank < int(np.ceil(1.5 * n / len(start)))]
    return cand[np.argsort(-resp[cand])[:n]]


def fill_features(frame: Frame, n_features: int = 2000) -> None:
    # Both images are extracted at once: OpenCV releases the GIL, and this sits on the
    # tracking path where it costs ~12 ms/frame otherwise.
    right = _pool.submit(extract_orb, frame.image_right, n_features, tag="R")
    frame.keypoints, frame.descriptors, frame.octave, frame.angle = extract_orb(frame.image_left, n_features)
    frame.points = np.full(len(frame.keypoints), None, dtype=object)
    frame.outlier = np.zeros(len(frame.keypoints), dtype=bool)
    pr, dr, octr, _ = right.result()
    match_stereo(frame, pr, dr, octr)


def match_stereo(frame: Frame, pr: np.ndarray, dr: np.ndarray, octr: np.ndarray) -> None:
    """ORB-SLAM2 ComputeStereoMatches: row-band descriptor match, then SAD subpixel refinement."""
    n = len(frame.keypoints)
    frame.u_right = np.full(n, np.nan)
    if n == 0 or len(pr) == 0:
        return
    order = np.argsort(pr[:, 1])
    pr, dr, octr = pr[order], dr[order], octr[order]

    ul, vl, octl = frame.keypoints[:, 0], frame.keypoints[:, 1], frame.octave
    band = 2.0 * SCALES[-1]
    i, j = ragged_pairs(np.searchsorted(pr[:, 1], vl - band), np.searchsorted(pr[:, 1], vl + band, "right"))
    disp = ul[i] - pr[j, 0]
    ok = (
        (np.abs(vl[i] - pr[j, 1]) <= 2.0 * SCALES[octr[j]])
        & (np.abs(octl[i] - octr[j]) <= 1)
        & (disp >= 0)
        & (disp <= frame.camera.fx)
    )
    i, j = i[ok], j[ok]
    li, k, best, _ = best_two(i, hamming(frame.descriptors[i], dr[j]))
    good = best < (TH_HIGH + TH_LOW) // 2
    li, rj = li[good], j[k[good]]

    ur, sad = _subpixel(frame.image_left, frame.image_right, frame.keypoints[li], pr[rj, 0], octl[li])
    d = ul[li] - ur
    keep = np.isfinite(ur) & (d >= 0) & (d < frame.camera.fx)
    ur = np.where(d <= 0, ul[li] - 0.01, ur)
    if keep.any():
        keep &= sad <= 1.5 * 1.4 * np.median(sad[keep])
    frame.u_right[li[keep]] = ur[keep]


def _subpixel(left, right, kl, ur0, octv):
    """1D SAD search on the keypoint's pyramid level plus a parabola fit."""
    ur = np.full(len(kl), np.nan)
    best = np.full(len(kl), np.inf)
    off = np.arange(-_W, _W + 1)
    shifts = np.arange(-_L, _L + 1)
    h0, w0 = left.shape
    for lvl in np.unique(octv):
        s = SCALES[lvl]
        size = (round(w0 / s), round(h0 / s))
        IL = left if lvl == 0 else cv2.resize(left, size, interpolation=cv2.INTER_LINEAR)
        IR = right if lvl == 0 else cv2.resize(right, size, interpolation=cv2.INTER_LINEAR)
        h, w = IL.shape
        m = np.flatnonzero(octv == lvl)
        cx = np.round(kl[m, 0] / s).astype(int)
        cy = np.round(kl[m, 1] / s).astype(int)
        cr = np.round(ur0[m] / s).astype(int)
        inb = (cy >= _W) & (cy + _W < h) & (cx >= _W) & (cx + _W < w) & (cr >= _L + _W) & (cr + _L + _W < w)
        m, cx, cy, cr = m[inb], cx[inb], cy[inb], cr[inb]
        if len(m) == 0:
            continue
        rows = cy[:, None, None] + off[None, :, None]
        pl = IL[rows, cx[:, None, None] + off[None, None, :]].astype(np.float32)
        pl -= pl[:, _W : _W + 1, _W : _W + 1]
        cols = cr[:, None, None, None] + shifts[None, :, None, None] + off[None, None, None, :]
        pr = IR[rows[:, None], cols].astype(np.float32)
        pr -= pr[:, :, _W : _W + 1, _W : _W + 1]
        sad = np.abs(pl[:, None] - pr).sum(axis=(2, 3))
        b = sad.argmin(axis=1)
        a = np.arange(len(m))
        bi = np.clip(b, 1, 2 * _L - 1)
        d1, d2, d3 = sad[a, bi - 1], sad[a, bi], sad[a, bi + 1]
        den = 2.0 * (d1 + d3 - 2.0 * d2)
        delta = np.divide(d1 - d3, den, out=np.zeros_like(den), where=den != 0)
        ok = (b > 0) & (b < 2 * _L) & (np.abs(delta) <= 1.0)
        ur[m[ok]] = s * (cr[ok] + shifts[b[ok]] + delta[ok])
        best[m[ok]] = sad[a, b][ok]
    return ur, best
