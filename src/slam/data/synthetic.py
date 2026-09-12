from __future__ import annotations

import numpy as np

from slam.data.frame import Frame, StereoCalib


def synthetic_stereo_sequence(
    n_frames: int = 12,
    n_points: int = 80,
    noise: float = 0.4,
    seed: int = 0,
) -> tuple[list[Frame], np.ndarray, list[np.ndarray]]:
    """Injected stereo keypoints for VO/BA tests without ORB."""
    rng = np.random.default_rng(seed)
    cam = StereoCalib(718.0, 718.0, 607.0, 185.0, 0.54, width=1241, height=376)
    Xw = np.column_stack(
        [
            rng.uniform(-8, 20, n_points),
            rng.uniform(-4, 4, n_points),
            rng.uniform(8, 30, n_points),
        ]
    )
    frames: list[Frame] = []
    gt: list[np.ndarray] = []
    dummy = np.zeros((cam.height, cam.width), np.uint8)
    for i in range(n_frames):
        T_wc = np.eye(4)
        T_wc[0, 3] = 0.8 * i
        T_wc[2, 3] = 0.05 * i
        yaw = 0.01 * i
        T_wc[:3, :3] = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ]
        )
        T_cw = np.linalg.inv(T_wc)
        Xc = (T_cw[:3, :3] @ Xw.T).T + T_cw[:3, 3]
        vis = Xc[:, 2] > 1.0
        uv = cam.project_stereo(Xc)
        vis &= cam.in_image(uv[:, :2], border=5)
        vis &= uv[:, 0] > uv[:, 2] + 1.0
        wids = np.flatnonzero(vis)
        kps = uv[vis, :2] + rng.normal(0, noise, size=(int(vis.sum()), 2))
        ur = uv[vis, 2] + rng.normal(0, noise, size=(int(vis.sum()),))
        desc = np.zeros((len(wids), 32), np.uint8)
        for row, wid in enumerate(wids):
            desc[row, :4] = np.frombuffer(np.int32(wid).tobytes(), np.uint8)
            desc[row, 4:] = (wid * 13 + np.arange(28, dtype=np.int32)) % 256
        f = Frame(
            frame_id=i,
            stamp=0.1 * i,
            image_left=dummy,
            image_right=dummy,
            camera=cam,
            T_cw=T_cw,
            keypoints=kps,
            descriptors=desc,
            u_right=ur,
            octave=np.zeros(len(kps), dtype=int),
            angle=np.zeros(len(kps)),
            points=np.full(len(kps), None, dtype=object),
            outlier=np.zeros(len(kps), dtype=bool),
            gt_T_cw=T_cw.copy(),
        )
        frames.append(f)
        gt.append(T_cw.copy())
    return frames, Xw, gt
