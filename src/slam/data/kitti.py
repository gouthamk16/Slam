from pathlib import Path

import cv2
import numpy as np

from slam.data.frame import Frame, StereoCalib


def load_times(path: Path) -> list[float]:
    return [float(line) for line in path.read_text().splitlines() if line.strip()]


def load_poses(path: Path) -> list[np.ndarray]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        T_wc = np.eye(4)
        T_wc[:3, :] = np.fromstring(line, sep=" ").reshape(3, 4)
        T_cw = np.linalg.inv(T_wc)
        out.append(T_cw)
    return out


def load_calib(path: Path) -> StereoCalib:
    vals = {}
    for line in path.read_text().splitlines():
        key, rest = line.split(":", 1)
        vals[key.strip()] = np.fromstring(rest, sep=" ")

    P0 = vals["P0"].reshape(3, 4)
    P1 = vals["P1"].reshape(3, 4)
    return StereoCalib(
        fx=float(P0[0, 0]),
        fy=float(P0[1, 1]),
        cx=float(P0[0, 2]),
        cy=float(P0[1, 2]),
        baseline=float(-P1[0, 3] / P1[0, 0]),
    )


class KittiSequence:
    def __init__(self, root: Path, seq: str = "00"):
        self.seq_dir = Path(root) / "sequences" / seq
        self.times = load_times(self.seq_dir / "times.txt")
        pose_file = Path(root) / "poses" / f"{seq}.txt"
        self.gt = load_poses(pose_file) if pose_file.exists() else []
        self.files = sorted((self.seq_dir / "image_0").glob("*.png"))
        self.camera = load_calib(self.seq_dir / "calib.txt")
        if self.files:
            im = cv2.imread(str(self.files[0]), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                self.camera.height, self.camera.width = int(im.shape[0]), int(im.shape[1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> Frame:
        name = self.files[i].name
        left = cv2.imread(str(self.files[i]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(self.seq_dir / "image_1" / name), cv2.IMREAD_GRAYSCALE)
        gt = self.gt[i] if i < len(self.gt) else None
        return Frame(
            frame_id=i,
            stamp=self.times[i] if i < len(self.times) else float(i) * 0.1,
            image_left=left,
            image_right=right,
            gt_T_cw=gt,
            camera=self.camera,
        )
