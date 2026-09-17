"""EuRoC MAV (ASL layout): rectified stereo frames, IMU samples and ground truth."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from slam.data.frame import Frame, StereoCalib
from slam.geometry.lie import invert


def _sensor(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _transform(entry: dict) -> np.ndarray:
    return np.array(entry["data"], dtype=np.float64).reshape(4, 4)


def _intrinsics(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    fu, fv, cu, cv = cam["intrinsics"]
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]])
    return K, np.array(cam["distortion_coefficients"], dtype=np.float64)


class EurocSequence:
    def __init__(self, root: Path):
        self.mav = mav = Path(root) / "mav0"
        cam0, cam1 = _sensor(mav / "cam0/sensor.yaml"), _sensor(mav / "cam1/sensor.yaml")
        self.fps = float(cam0["rate_hz"])
        w, h = cam0["resolution"]
        K0, D0 = _intrinsics(cam0)
        K1, D1 = _intrinsics(cam1)
        T_B_C0 = _transform(cam0["T_BS"])  # T_BS maps sensor coordinates into the body (= IMU) frame
        T_C1_C0 = invert(_transform(cam1["T_BS"])) @ T_B_C0
        R1, R2, P1, P2, *_ = cv2.stereoRectify(
            K0, D0, K1, D1, (w, h), T_C1_C0[:3, :3].copy(), T_C1_C0[:3, 3:].copy(), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        self._maps = (
            cv2.initUndistortRectifyMap(K0, D0, R1, P1, (w, h), cv2.CV_16SC2),
            cv2.initUndistortRectifyMap(K1, D1, R2, P2, (w, h), cv2.CV_16SC2),
        )
        self.camera = StereoCalib(
            fx=P1[0, 0], fy=P1[1, 1], cx=P1[0, 2], cy=P1[1, 2], baseline=-P2[0, 3] / P2[0, 0], width=w, height=h
        )
        # Rectification rotates cam0 (X_rect = R1 X_cam0). Leaving R1 out of body-from-camera would
        # silently rotate every pose compared against body-frame ground truth.
        T_C0_rect = np.eye(4)
        T_C0_rect[:3, :3] = R1.T
        self.T_BC = T_B_C0 @ T_C0_rect

        # Some sequences drop frames on one camera (V2_03: 1922 left, 2336 right); keep only stereo pairs.
        stamps = [np.loadtxt(mav / c / "data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64) for c in ("cam0", "cam1")]
        self.stamps_ns = np.intersect1d(*stamps)
        self.names = np.char.add(self.stamps_ns.astype(str), ".png")

        imu = mav / "imu0/data.csv"
        self.imu_t = np.loadtxt(imu, delimiter=",", comments="#", usecols=0, dtype=np.int64) * 1e-9
        vals = np.loadtxt(imu, delimiter=",", comments="#", usecols=range(1, 7))
        self.imu_gyro, self.imu_acc = vals[:, :3], vals[:, 3:]
        self.imu_noise = {k: v for k, v in _sensor(mav / "imu0/sensor.yaml").items() if "noise" in k or "walk" in k}

        self.gt = self._ground_truth(mav / "state_groundtruth_estimate0/data.csv")

    def _ground_truth(self, path: Path) -> list[np.ndarray | None]:
        """World-to-rectified-camera pose per frame from the nearest 200 Hz ground-truth sample."""
        if not path.exists():
            return [None] * len(self.names)
        t = np.loadtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64)
        g = np.loadtxt(path, delimiter=",", comments="#", usecols=range(1, 8))
        R = Rotation.from_quat(g[:, [4, 5, 6, 3]]).as_matrix()  # file order is w, x, y, z
        k = np.clip(np.searchsorted(t, self.stamps_ns), 1, len(t) - 1)
        k = np.where(np.abs(t[k - 1] - self.stamps_ns) < np.abs(t[k] - self.stamps_ns), k - 1, k)
        near = np.abs(t[k] - self.stamps_ns) < 5_000_000  # ground truth can start after the images
        out: list[np.ndarray | None] = []
        for kk, ok in zip(k, near):
            if not ok:
                out.append(None)
                continue
            T_WB = np.eye(4)
            T_WB[:3, :3], T_WB[:3, 3] = R[kk], g[kk, :3]
            out.append(invert(T_WB @ self.T_BC))
        return out

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> Frame:
        raw = [cv2.imread(str(self.mav / c / "data" / self.names[i]), cv2.IMREAD_GRAYSCALE) for c in ("cam0", "cam1")]
        left, right = (cv2.remap(im, m1, m2, cv2.INTER_LINEAR) for im, (m1, m2) in zip(raw, self._maps))
        return Frame(
            frame_id=i, stamp=self.stamps_ns[i] * 1e-9, image_left=left, image_right=right,
            gt_T_cw=self.gt[i], camera=self.camera,
        )
