"""EuRoC: rectification must line up rows, and image motion must agree with ground truth.

The rotation check also catches a wrong T_BS direction: cam0's extrinsic is a ~90 degree rotation,
so inverting it would conjugate every relative rotation by 90 degrees. Rotation comes from stereo
PnP, not the essential matrix: indoor motion is close to pure rotation, where E is ill-conditioned.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from slam.data.euroc import EurocSequence
from slam.geometry.lie import invert, log_so3

ROOT = Path("datasets/euroc/V1_01_easy")
pytestmark = pytest.mark.skipif(not (ROOT / "mav0").exists(), reason="EuRoC V1_01_easy not downloaded")


@pytest.fixture(scope="module")
def seq():
    return EurocSequence(ROOT)


def _matches(a, b):
    orb = cv2.ORB_create(2000)
    k0, d0 = orb.detectAndCompute(a, None)
    k1, d1 = orb.detectAndCompute(b, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    return np.float64([k0[x.queryIdx].pt for x in m]), np.float64([k1[x.trainIdx].pt for x in m])


def test_rectified_rows_line_up(seq):
    f = seq[600]
    pl, pr = _matches(f.image_left, f.image_right)
    dy = pl[:, 1] - pr[:, 1]
    good = np.abs(dy) < 5
    assert good.mean() > 0.6
    assert np.median(np.abs(dy[good])) < 0.6
    assert seq.camera.baseline == pytest.approx(0.11, abs=0.01)


def _pnp_rotation(seq, a, b):
    pl, pr = _matches(a.image_left, a.image_right)
    ok = (np.abs(pl[:, 1] - pr[:, 1]) < 1) & (pl[:, 0] - pr[:, 0] > 1)
    X = seq.camera.triangulate_stereo(pl[ok, 0], pl[ok, 1], pr[ok, 0])
    p0, p1 = _matches(a.image_left, b.image_left)
    d = np.linalg.norm(p0[:, None] - pl[ok][None], axis=2)
    near = d.min(1) < 0.5
    _, rvec, _, _ = cv2.solvePnPRansac(
        X[d.argmin(1)[near]], p1[near], seq.camera.K, None, reprojectionError=1.5, iterationsCount=500
    )
    return cv2.Rodrigues(rvec)[0]


def test_image_rotation_matches_gt(seq):
    errs = []
    for i in range(300, 2700, 300):
        a, b = seq[i], seq[i + 5]
        R_gt = (b.gt_T_cw @ invert(a.gt_T_cw))[:3, :3]
        errs.append(np.degrees(np.linalg.norm(log_so3(_pnp_rotation(seq, a, b) @ R_gt.T))))
    assert np.median(errs) < 0.4, f"image rotation disagrees with ground truth: {np.round(errs, 2)} deg"
