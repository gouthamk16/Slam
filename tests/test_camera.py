import numpy as np
from pathlib import Path

from slam.data.frame import StereoCalib
from slam.data.kitti import KittiSequence


def _kitti_cam() -> StereoCalib:
    return KittiSequence(Path("datasets/kitti"), "00")[0].camera


def test_optical_axis_hits_principal_point():
    cam = _kitti_cam()
    uv = cam.project(np.array([[0.0, 0.0, 10.0]]))
    np.testing.assert_allclose(uv[0], [cam.cx, cam.cy])


def test_project_unproject_roundtrip():
    cam = _kitti_cam()
    Xc = np.array([[0.5, -0.2, 8.0], [1.0, 0.4, 12.0]])
    uv = cam.project(Xc)
    X2 = cam.unproject(uv, Xc[:, 2])
    np.testing.assert_allclose(X2, Xc, atol=1e-6)


def test_stereo_depth_from_disparity():
    cam = StereoCalib(718.0, 718.0, 607.0, 185.0, 0.54, 1241, 376)
    Xc = np.array([[1.0, 0.0, 10.0]])
    uvlr = cam.project_stereo(Xc)[0]
    depth = cam.depth_from_disparity(uvlr[0], uvlr[2])
    np.testing.assert_allclose(depth, 10.0, atol=1e-6)
    X2 = cam.triangulate_stereo(np.array([uvlr[0]]), np.array([uvlr[1]]), np.array([uvlr[2]]))
    np.testing.assert_allclose(X2[0], Xc[0], atol=1e-6)


def test_in_image():
    cam = StereoCalib(718.0, 718.0, 607.0, 185.0, 0.54, 1241, 376)
    uv = np.array([[10, 10], [-1, 10], [2000, 10]])
    m = cam.in_image(uv)
    assert list(m) == [True, False, False]
