from pathlib import Path

import cv2
import numpy as np

from slam.data.kitti import KittiSequence
from slam.features.orb import fill_features


def test_stereo_matches_agree_with_sgbm_on_kitti_frame0():
    f = KittiSequence(Path("datasets/kitti"), "00")[0]
    fill_features(f, 2000)
    ok = np.isfinite(f.u_right)
    assert ok.sum() >= 1000

    sgbm = cv2.StereoSGBM_create(0, 128, 5, 8 * 25, 32 * 25, 2, 63, 8, 80, 2, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    ref = sgbm.compute(f.image_left, f.image_right).astype(np.float32) / 16.0
    uv = np.round(f.keypoints[ok]).astype(int)
    ref = ref[uv[:, 1], uv[:, 0]]
    disp = f.keypoints[ok, 0] - f.u_right[ok]
    valid = ref > 1.0
    assert np.median(np.abs(disp[valid] - ref[valid]) / ref[valid]) < 0.03
