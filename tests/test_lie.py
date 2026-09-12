import numpy as np

from slam.geometry.lie import exp_se3, exp_so3, invert, log_se3, log_so3, transform_points


def test_so3_roundtrip():
    rng = np.random.default_rng(1)
    w = rng.normal(size=3) * 0.4
    R = exp_so3(w)
    w2 = log_so3(R)
    np.testing.assert_allclose(exp_so3(w2), R, atol=1e-8)


def test_se3_roundtrip():
    rng = np.random.default_rng(2)
    xi = rng.normal(size=6) * 0.3
    T = exp_se3(xi)
    np.testing.assert_allclose(exp_se3(log_se3(T)), T, atol=1e-8)


def test_invert_is_inverse():
    T = exp_se3([0.1, -0.2, 0.05, 1.0, 2.0, -0.5])
    I = invert(T) @ T
    np.testing.assert_allclose(I, np.eye(4), atol=1e-10)


def test_transform_points_matches_matrix():
    T = exp_se3([0, 0.2, 0, 1, 0, 3])
    X = np.array([[1.0, 2.0, 4.0], [0, 0, 5]])
    Y = transform_points(T, X)
    homog = np.c_[X, np.ones(2)].T
    Ym = (T @ homog)[:3].T
    np.testing.assert_allclose(Y, Ym)
