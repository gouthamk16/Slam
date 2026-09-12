import pytest

from slam.data.synthetic import synthetic_stereo_sequence
from slam.eval import ate_rmse
from slam.system import Config, SlamSystem


@pytest.mark.parametrize("threaded", [False, True])
def test_synthetic_stereo_tracking(threaded):
    frames, _, gt = synthetic_stereo_sequence(n_frames=20, n_points=1500, noise=0.3, seed=3)
    slam = SlamSystem(Config(vocab=None, threaded=threaded))
    for f in frames:
        slam.track(f)
    slam.close()
    assert slam.state == "ok"
    assert len(slam.map.keyframes) >= 2
    err = ate_rmse(slam.trajectory(), gt)
    assert err < 0.05, f"ATE {err:.3f} m"
