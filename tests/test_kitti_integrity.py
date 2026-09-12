"""Image motion must agree with the GT pose file.

A mislabeled mirror once spliced seq 01 into seq 00; stereo pairs stayed consistent,
so only a check against GT motion catches it.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from slam.data.kitti import KittiSequence
from slam.geometry.lie import invert, log_so3


@pytest.fixture(scope="module")
def seq():
    return KittiSequence(Path("datasets/kitti"), "00")


@pytest.mark.parametrize("i", [0, 500, 1100, 2000, 4000])
def test_image_rotation_matches_gt(seq, i):
    a, b = seq[i], seq[i + 1]
    orb = cv2.ORB_create(2000)
    k0, d0 = orb.detectAndCompute(a.image_left, None)
    k1, d1 = orb.detectAndCompute(b.image_left, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    p0 = np.float64([k0[x.queryIdx].pt for x in m])
    p1 = np.float64([k1[x.trainIdx].pt for x in m])
    K = seq.camera.K
    E, mask = cv2.findEssentialMat(p0, p1, K, cv2.RANSAC, 0.999, 1.0)
    _, R, _, _ = cv2.recoverPose(E, p0, p1, K, mask=mask)
    R_gt = (b.gt_T_cw @ invert(a.gt_T_cw))[:3, :3]
    err = np.degrees(np.linalg.norm(log_so3(R @ R_gt.T)))
    assert err < 0.5, f"frame {i}: image rotation disagrees with GT by {err:.2f} deg"
