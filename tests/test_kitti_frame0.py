from pathlib import Path

import numpy as np

from slam.data.kitti import KittiSequence


def test_stereo_frame0_loads():
    seq = KittiSequence(Path("datasets/kitti"), "00")
    assert len(seq) == 4541
    frame = seq[0]
    assert frame.frame_id == 0
    assert frame.stamp == 0.0
    assert frame.image_left is not None
    assert frame.image_right is not None
    assert frame.image_left.shape == (376, 1241)
    assert frame.image_right.shape == frame.image_left.shape
    assert frame.image_left.dtype == frame.image_right.dtype
    assert frame.gt_T_cw is not None
    np.testing.assert_allclose(frame.gt_T_cw, np.eye(4), atol=1e-8)
    assert frame.camera is not None
    assert abs(frame.camera.fx - 718.856) < 1e-3
    assert abs(frame.camera.baseline - 0.53716) < 1e-4

