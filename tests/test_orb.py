import numpy as np

from slam.features.orb import extract_orb


def test_orb_extracts_corners():
    img = np.zeros((240, 320), np.uint8)
    rng = np.random.default_rng(0)
    img[:] = rng.integers(0, 50, img.shape, dtype=np.uint8)
    for _ in range(40):
        y, x = rng.integers(20, 220), rng.integers(20, 300)
        img[y : y + 4, x : x + 4] = 255
    pts, desc, octave, _ = extract_orb(img, n_features=500)
    assert len(pts) > 10
    assert desc.shape[1] == 32
    assert len(octave) == len(pts) and octave.max() < 8
