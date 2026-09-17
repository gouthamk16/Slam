#!/usr/bin/env python3
"""Fetch EuRoC MAV sequences (ASL layout) from a Hugging Face mirror.

The mirror nests one zip per sequence inside three large archives. Each sequence is pulled out with
HTTP range requests, so a ~1 GB sequence costs ~1 GB, not the 25 GB of the archives. ETH's own host
(robotics.ethz.ch) was unreachable when this was written.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from remotezip import RemoteZip

MIRROR = "https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/resolve/main"
ARCHIVE = {"MH": "machine_hall", "V1": "vicon_room1", "V2": "vicon_room2"}
SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]


def fetch(seq: str, root: Path) -> None:
    out = root / seq
    if (out / "mav0").exists():
        print(f"{seq}: already extracted")
        return
    zpath = root / ".downloads" / f"{seq}.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if not zpath.exists():
        arc = ARCHIVE[seq[:2]]
        print(f"{seq}: downloading from {arc}.zip", flush=True)
        tmp = zpath.with_suffix(".part")
        with RemoteZip(f"{MIRROR}/{arc}.zip") as z, z.open(f"{arc}/{seq}/{seq}.zip") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 << 20)
        tmp.rename(zpath)
    print(f"{seq}: extracting", flush=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out, [n for n in z.namelist() if n.startswith("mav0/") and ".DS_Store" not in n])
    print(f"{seq}: done -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("seqs", nargs="+", help=f"sequence names or 'all' ({', '.join(SEQUENCES)})")
    p.add_argument("--root", default="datasets/euroc")
    args = p.parse_args()
    for seq in SEQUENCES if args.seqs == ["all"] else args.seqs:
        if seq not in SEQUENCES:
            raise SystemExit(f"unknown sequence {seq}")
        fetch(seq, Path(args.root))


if __name__ == "__main__":
    main()
