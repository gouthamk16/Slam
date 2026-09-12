from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StereoCalib:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float
    width: int = 1241
    height: int = 376

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def project(self, Xc: np.ndarray) -> np.ndarray:
        Xc = np.asarray(Xc, dtype=np.float64).reshape(-1, 3)
        z = np.maximum(Xc[:, 2], 1e-12)
        uv = np.empty((len(Xc), 2), dtype=np.float64)
        uv[:, 0] = self.fx * Xc[:, 0] / z + self.cx
        uv[:, 1] = self.fy * Xc[:, 1] / z + self.cy
        return uv

    def unproject(self, uv: np.ndarray, depth: np.ndarray | float) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        d = np.asarray(depth, dtype=np.float64).reshape(-1)
        if d.size == 1:
            d = np.repeat(d, len(uv))
        x = (uv[:, 0] - self.cx) / self.fx * d
        y = (uv[:, 1] - self.cy) / self.fy * d
        return np.stack([x, y, d], axis=1)

    def project_stereo(self, Xc: np.ndarray) -> np.ndarray:
        uv = self.project(Xc)
        Xc = np.asarray(Xc, dtype=np.float64).reshape(-1, 3)
        z = np.maximum(Xc[:, 2], 1e-12)
        ur = self.fx * (Xc[:, 0] - self.baseline) / z + self.cx
        return np.column_stack([uv, ur])

    def depth_from_disparity(self, u_l: np.ndarray, u_r: np.ndarray) -> np.ndarray:
        disp = np.asarray(u_l, dtype=np.float64) - np.asarray(u_r, dtype=np.float64)
        disp = np.maximum(disp, 1e-4)
        return self.fx * self.baseline / disp

    def triangulate_stereo(self, u_l: np.ndarray, v: np.ndarray, u_r: np.ndarray) -> np.ndarray:
        d = self.depth_from_disparity(u_l, u_r)
        return self.unproject(np.column_stack([u_l, v]), d)

    def in_image(self, uv: np.ndarray, border: int = 0) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        return (
            (uv[:, 0] >= border)
            & (uv[:, 0] < self.width - border)
            & (uv[:, 1] >= border)
            & (uv[:, 1] < self.height - border)
        )


@dataclass
class Frame:
    frame_id: int
    stamp: float
    image_left: np.ndarray
    image_right: np.ndarray | None = None
    gt_T_cw: np.ndarray | None = None
    camera: StereoCalib | None = None
    T_cw: np.ndarray = field(default_factory=lambda: np.eye(4))
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    descriptors: np.ndarray | None = None
    octave: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    angle: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    u_right: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    points: np.ndarray = field(default_factory=lambda: np.full((0,), None, dtype=object))  # MapPoint per keypoint
    outlier: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
