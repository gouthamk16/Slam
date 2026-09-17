"""Live view in Rerun: camera image with tracked features, map points, keyframes, trajectories."""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from slam.eval import centers
from slam.geometry.align import horn_se3
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
        # Sent every run: the viewer otherwise restores the last session's 3D eye, which after a
        # KITTI run sits hundreds of metres from a 3 m EuRoC room.
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    rrb.Spatial3DView(origin="world"),
                    rrb.Vertical(rrb.Spatial2DView(origin="world/camera/image"), rrb.TimeSeriesView(origin="stats")),
                ),
                rrb.TimePanel(timeline="frame", play_state="following"),
            )
        )
        self.every = every
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log(
            "world/camera/image",
            rr.Pinhole(image_from_camera=camera.K, resolution=[camera.width, camera.height]),
            static=True,
        )
        self.gt = gt or []  # per frame, None where there is no ground truth
        self.gt_frames = np.array([i for i, g in enumerate(self.gt) if g is not None], dtype=int)
        self.gt_centers = centers([self.gt[i] for i in self.gt_frames]) if len(self.gt_frames) else None

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
            self._log_map(slam, frame.frame_id)

    def _log_map(self, slam, frame_id: int) -> None:
        # Static entities keep only the latest map, so memory stays bounded over long runs.
        with slam.map.lock:
            pts = list(slam.map.points)
            kfs = list(slam.map.keyframes.values())
            traj = slam.trajectory()
        if pts:
            gray = (np.array([mp.color for mp in pts]) * 255).astype(np.uint8)
            rr.log(
                "world/map",
                rr.Points3D(np.stack([mp.Xw for mp in pts]), colors=np.c_[gray, gray, gray], radii=rr.Radius.ui_points(1.5)),
                static=True,
            )
        if kfs:
            rr.log("world/keyframes", rr.Points3D(np.stack([k.center for k in kfs]), colors=(80, 160, 255), radii=rr.Radius.ui_points(4.0)), static=True)
        if len(traj) > 1:
            rr.log("world/trajectory", rr.LineStrips3D([centers(traj)], colors=(40, 220, 40)), static=True)
        T = self._gt_alignment(traj, frame_id)
        if T is not None:
            seen = self.gt_centers[: np.searchsorted(self.gt_frames, frame_id, "right")]  # not the flight still to come
            rr.log("world/gt", rr.LineStrips3D([seen @ T[:3, :3].T + T[:3, 3]], colors=(150, 150, 150)), static=True)

    def _gt_alignment(self, traj, frame_id: int):
        """Ground-truth world -> SLAM world, fitted to the frames tracked so far.

        SLAM's world is the camera at initialisation, and ground truth may start later (EuRoC), so a
        fixed anchor is wrong. Horn over all shared frames matches how ATE is scored; until the path
        spreads out enough for that to be well posed, anchor on the first shared frame.
        """
        first = frame_id + 1 - len(traj)
        pairs = [(e, g) for e, g in zip(traj, self.gt[first : frame_id + 1]) if g is not None]
        if not pairs:
            return None
        gt = centers([g for _, g in pairs])
        if len(gt) >= 3 and np.linalg.svd(gt - gt.mean(axis=0), compute_uv=False)[1] / np.sqrt(len(gt)) > 0.2:
            return horn_se3(gt, centers([e for e, _ in pairs]))
        e, g = pairs[0]
        return invert(e) @ g
