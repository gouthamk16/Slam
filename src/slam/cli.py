"""CLI: python -m slam --seq 00   |   python -m slam --dataset euroc --seq V1_01_easy"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from slam.eval import ate_rmse, kitti_rpe, write_kitti_poses
from slam.system import Config, SlamSystem


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stereo visual SLAM on KITTI odometry or EuRoC MAV")
    p.add_argument("--dataset", choices=("kitti", "euroc"), default="kitti")
    p.add_argument("--root", default=None, help="dataset root (default datasets/<dataset>)")
    p.add_argument("--seq", default="00")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    p.add_argument("--no-viz", action="store_true")
    p.add_argument("--rrd", default=None, help="save a Rerun recording instead of spawning the viewer")
    p.add_argument("--sync", action="store_true", help="run mapping and loop closing inline (deterministic, slower)")
    args = p.parse_args(argv)
    root = Path(args.root or f"datasets/{args.dataset}")

    if args.dataset == "euroc":
        from slam.data.euroc import EurocSequence

        seq = EurocSequence(root / args.seq)
        # ORB-SLAM2/3 EuRoC settings: 1200 features on 752x480
        cfg = Config(n_features=1200, fps=round(seq.fps), threaded=not args.sync)
    else:
        from slam.data.kitti import KittiSequence

        seq = KittiSequence(root, args.seq)
        cfg = Config(threaded=not args.sync)
    n = len(seq) if args.max_frames <= 0 else min(args.max_frames, len(seq))
    gt = list(seq.gt[:n]) + [None] * (n - len(seq.gt[:n]))
    slam = SlamSystem(cfg)
    viewer = None
    if not args.no_viz:
        from slam.viz import Viewer

        viewer = Viewer(seq.camera, gt, args.rrd)

    print(f"{args.dataset} {args.seq}: {n} frames")
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
    pairs = [(e, g) for e, g in zip(traj, gt[n - len(traj) :]) if g is not None]
    if pairs:
        est, ref = map(list, zip(*pairs))
        msg = f"ATE {ate_rmse(est, ref):.3f} m over {len(pairs)} frames"
        if args.dataset == "kitti":
            t_rel, r_rel = kitti_rpe(est, ref)
            msg += f"   t_rel {t_rel:.2f} %   r_rel {r_rel:.3f} deg/100m"
        print(msg)
