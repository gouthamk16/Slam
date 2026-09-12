"""DBoW2 vocabulary tree (ORB-SLAM's ORBvoc.txt): words, TF-IDF bag of words, L1 score."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from slam.features.matcher import hamming

LEVELS_UP = 4  # direct-index node depth = L - 4, as ORB-SLAM's transform(..., 4)


class Vocabulary:
    def __init__(self, path: str | Path):
        path = Path(path)
        cache = path.with_suffix(".npz")
        if not cache.exists():
            _convert(path, cache)
        d = np.load(cache)
        self.k, self.L = int(d["k"]), int(d["L"])
        self.desc = d["desc"]  # (n_nodes, 32), node 0 is the root
        self.weight = d["weight"]  # idf weight per node (leaves only matter)
        self.word = d["word"]  # word id per node, -1 for inner nodes
        self.child_start = d["child_start"]
        self.child_count = d["child_count"]
        self.children = d["children"]  # node ids grouped by parent
        self.leaf = np.flatnonzero(self.word >= 0)  # word id -> node id

    def transform(self, desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Word id and direct-index node id for each descriptor (greedy tree descent)."""
        n = len(desc)
        cur = np.zeros(n, dtype=np.int64)
        node = np.zeros(n, dtype=np.int64)
        ar = np.arange(self.k)
        for level in range(self.L):
            cnt = self.child_count[cur]
            if not cnt.any():
                break
            slots = np.minimum(self.child_start[cur, None] + ar, len(self.children) - 1)
            kids = self.children[slots]
            d = np.where(ar < cnt[:, None], hamming(desc[:, None, :], self.desc[kids]), 1 << 20)
            cur = np.where(cnt > 0, kids[np.arange(n), d.argmin(axis=1)], cur)
            if level + 1 == self.L - LEVELS_UP:
                node = cur.copy()
        return self.word[cur], node

    def bow(self, words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """L1-normalized TF-IDF vector as (sorted word ids, weights)."""
        ids, cnt = np.unique(words[words >= 0], return_counts=True)
        w = cnt * self.weight[self.leaf[ids]]
        keep = w > 0
        return ids[keep], w[keep] / max(w[keep].sum(), 1e-12)

    def compute(self, kf) -> None:
        words, kf.node = self.transform(kf.descriptors)
        kf.bow = self.bow(words)


def score(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """DBoW2 L1 score in [0, 1]."""
    _, i, j = np.intersect1d(a[0], b[0], assume_unique=True, return_indices=True)
    x, y = a[1][i], b[1][j]
    return float(0.5 * (x + y - np.abs(x - y)).sum())


def _convert(txt: Path, out: Path) -> None:
    """Parse DBoW2 text format: header 'k L scoring weighting', then per node
    'parent is_leaf d0..d31 weight' in id order (the root, id 0, is implicit)."""
    with open(txt) as f:
        k, L = (int(x) for x in f.readline().split()[:2])
    rows = np.loadtxt(txt, skiprows=1, dtype=np.float64)
    n = len(rows) + 1
    parent = np.r_[-1, rows[:, 0].astype(np.int64)]
    leaf = np.r_[False, rows[:, 1] > 0]
    desc = np.zeros((n, 32), dtype=np.uint8)
    desc[1:] = rows[:, 2:34].astype(np.uint8)
    weight = np.r_[0.0, rows[:, 34]]
    word = np.full(n, -1, dtype=np.int64)
    word[leaf] = np.arange(leaf.sum())
    children = np.argsort(parent[1:], kind="stable") + 1
    child_count = np.bincount(parent[1:], minlength=n)
    child_start = np.r_[0, np.cumsum(child_count)[:-1]]
    np.savez(out, k=k, L=L, desc=desc, weight=weight, word=word,
             child_start=child_start, child_count=child_count, children=children)
