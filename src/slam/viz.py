"""Live view in Rerun: camera image with tracked features, map points, keyframes, trajectories."""

from __future__ import annotations

import numpy as np
import rerun as rr

from slam.eval import centers
from slam.geometry.lie import invert


class Viewer:
    def __init__(self, camera, gt: list[np.ndarray] | None = None, rrd: str | None = None, every: int = 10):
        rr.init("slam2.0")
        if rrd is None:
            try:
                rr.spawn()
            except RuntimeError as e:  # viewer binary not on PATH: record instead of failing the run
                rrd = "outputs/run.rrd"
                print(f"{e}\nlogging to {rrd} instead; open it with `.venv/bin/rerun {rrd}`")
        if rrd:
            rr.save(rrd)
        self.every = every
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log(
            "world/camera/image",
            rr.Pinhole(image_from_camera=camera.K, resolution=[camera.width, camera.height]),
            static=True,
        )
        if gt:
            rr.log("world/gt", rr.LineStrips3D([centers(gt)], colors=(150, 150, 150)), static=True)

    def update(self, frame, slam) -> None:
        rr.set_time("frame", sequence=frame.frame_id)
        T_wc = invert(frame.T_cw)
        rr.log("world/camera", rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]))
        rr.log("world/camera/image", rr.Image(frame.image_left).compress(jpeg_quality=80))
        tracked = np.not_equal(frame.points, None) & ~frame.outlier
        colors = np.where(tracked[:, None], (40, 220, 40), (210, 210, 210)).astype(np.uint8)
        rr.log("world/camera/image/features", rr.Points2D(frame.keypoints, colors=colors, radii=1.5))
        rr.log("stats/tracked", rr.Scalars(int(tracked.sum())))
        if frame.frame_id % self.every == 0:
            self._log_map(slam)

    def _log_map(self, slam) -> None:
        # Static entities keep only the latest map, so memory stays bounded over long runs.
        with slam.map.lock:
            pts = list(slam.map.points)
            kfs = list(slam.map.keyframes.values())
            traj = slam.trajectory()
        if pts:
            gray = (np.array([mp.color for mp in pts]) * 255).astype(np.uint8)
            rr.log(
                "world/map",
                rr.Points3D(np.stack([mp.Xw for mp in pts]), colors=np.c_[gray, gray, gray], radii=0.06),
                static=True,
            )
        if kfs:
            rr.log("world/keyframes", rr.Points3D(np.stack([k.center for k in kfs]), colors=(80, 160, 255), radii=0.25), static=True)
        if len(traj) > 1:
            rr.log("world/trajectory", rr.LineStrips3D([centers(traj)], colors=(40, 220, 40)), static=True)
