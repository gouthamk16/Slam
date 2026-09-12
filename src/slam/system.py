"""SLAM system: wires tracking, local mapping and loop closing together."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slam.data.frame import Frame
from slam.features.orb import fill_features
from slam.features.vocab import Vocabulary
from slam.loop import LoopCloser
from slam.map.structure import Map
from slam.mapping import LocalMapper
from slam.tracking import TrackCfg, Tracker


@dataclass
class Config:
    n_features: int = 2000
    th_depth: float = 35.0
    fps: int = 10
    vocab: str | None = "datasets/vocab/ORBvoc.txt"
    threaded: bool = False  # local mapping and loop closing on their own threads


class SlamSystem:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.map = Map()
        self.vocab = Vocabulary(self.cfg.vocab) if self.cfg.vocab else None
        self.mapper = LocalMapper(self.map, self.vocab, self.cfg.th_depth, self.cfg.threaded)
        self.loop = LoopCloser(self.map, self.cfg.threaded) if self.vocab else None
        self.mapper.loop = self.loop
        self.tracker = Tracker(self.map, self.mapper, TrackCfg(th_depth=self.cfg.th_depth, max_frames=self.cfg.fps))

    @property
    def state(self) -> str:
        return self.tracker.state

    def track(self, frame: Frame) -> np.ndarray:
        if len(frame.keypoints) == 0:
            fill_features(frame, self.cfg.n_features)
        return self.tracker.track(frame)

    def trajectory(self) -> list[np.ndarray]:
        with self.map.lock:
            return self.tracker.trajectory()

    def close(self) -> None:
        """Finish queued keyframes and stop worker threads (mapping feeds loop closing)."""
        self.mapper.close()
        if self.loop is not None:
            self.loop.close()
