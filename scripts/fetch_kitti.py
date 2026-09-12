#!/usr/bin/env python3
"""Fetch one KITTI odometry gray sequence from the official zip using HTTP range requests.

The full zip is 23 GB; one sequence is ~2.4 GB, so we read only its members.
"""

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from remotezip import RemoteZip

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip"
_local = threading.local()


def _zip() -> RemoteZip:
    # RemoteZip holds a single HTTP session and file cursor, so one per thread.
    if not hasattr(_local, "z"):
        _local.z = RemoteZip(URL)
    return _local.z


def _fetch(job: tuple[str, Path]) -> None:
    name, dst = job
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(_zip().read(name))
    tmp.rename(dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="00")
    p.add_argument("--root", default="datasets/kitti")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    prefix = f"dataset/sequences/{args.seq}/"
    out = Path(args.root) / "sequences" / args.seq
    with RemoteZip(URL) as z:
        names = [n for n in z.namelist() if n.startswith(prefix) and n.endswith(".png")]
    if not names:
        raise SystemExit(f"no members under {prefix}")

    todo = []
    for n in names:
        dst = out / n[len(prefix):]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            todo.append((n, dst))
    print(f"seq {args.seq}: {len(names)} images, {len(todo)} to fetch")

    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(_fetch, todo), 1):
            if i % 500 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
