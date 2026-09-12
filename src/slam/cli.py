"""CLI: python -m slam --seq 00"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from slam.data.kitti import KittiSequence
from slam.eval import ate_rmse, kitti_rpe, write_kitti_poses
from slam.system import Config, SlamSystem


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="KITTI stereo visual SLAM")
    p.add_argument("--root", default="datasets/kitti", help="KITTI odometry root")
    p.add_argument("--seq", default="00")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--rrd", default=None, help="save a Rerun recording instead of spawning the viewer")
    p.add_argument("--sync", action="store_true", help="run mapping and loop closing inline (deterministic, slower)")
    args = p.parse_args(argv)

    seq = KittiSequence(Path(args.root), args.seq)
    n = len(seq) if args.max_frames <= 0 else min(args.max_frames, len(seq))
    slam = SlamSystem(Config(threaded=not args.sync))
    viewer = None
    if not args.no_viz:
        from slam.viz import Viewer

        viewer = Viewer(seq.camera, seq.gt[:n], args.rrd)

    print(f"KITTI {args.seq}: {n} frames")
    lost = 0
    t0 = time.perf_counter()
    for i in range(n):
        f = seq[i]
        slam.track(f)
        lost += slam.state == "lost"
        if viewer is not None:
            viewer.update(f, slam)
        if i % 100 == 0:
            print(
                f"frame {i:5d}  {slam.state:4s}  kfs={len(slam.map.keyframes):4d}  "
                f"points={len(slam.map.points):6d}  inliers={slam.tracker.n_inliers:4d}  "
                f"{1000 * (time.perf_counter() - t0) / (i + 1):.0f} ms/frame",
                flush=True,
            )
    fps = n / (time.perf_counter() - t0)
    slam.close()

    traj = slam.trajectory()
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    path = out / f"traj_{args.seq}.txt"
    write_kitti_poses(path, traj)
    print(f"{fps:.1f} fps   lost frames {lost}   wrote {path}")
    if slam.loop is not None:
        print(f"loops closed (frame, matched frame): {slam.loop.loops}")
    gt = seq.gt[n - len(traj) : n]
    if gt:
        t_rel, r_rel = kitti_rpe(traj, gt)
        print(f"ATE {ate_rmse(traj, gt):.2f} m   t_rel {t_rel:.2f} %   r_rel {r_rel:.3f} deg/100m")
