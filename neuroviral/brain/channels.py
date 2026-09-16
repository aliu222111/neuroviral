"""Fuse extracted features into the five neural-channel activation time series.

This is the linear mapping-table from the spec made concrete. Each channel is a
clamped weighted sum of its driver features (missing features fall back to their
neutral baseline), then scaled by the content-type profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..utils import TimeGrid, clamp01, smooth
from . import weights as W

CHANNELS = ("reward", "attention", "emotion", "aversion", "language")


@dataclass
class Channels:
    grid: TimeGrid
    reward: np.ndarray
    attention: np.ndarray
    emotion: np.ndarray
    aversion: np.ndarray
    language: np.ndarray
    content_type: str = "auto"
    notes: List[str] = field(default_factory=list)

    @property
    def times(self) -> np.ndarray:
        return self.grid.times

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {c: getattr(self, c) for c in CHANNELS}


def _feature(features: Dict[str, np.ndarray], name: str, grid: TimeGrid, baseline: Dict[str, float]) -> np.ndarray:
    arr = features.get(name)
    if arr is None:
        return grid.full(float(baseline.get(name, 0.0)))
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] != grid.n:
        # length mismatch -> resize by clip/pad with baseline
        out = grid.full(float(baseline.get(name, 0.0)))
        m = min(arr.shape[0], grid.n)
        out[:m] = arr[:m]
        return out
    return arr


def build_channels(
    features: Dict[str, np.ndarray],
    grid: TimeGrid,
    content_type: str = "auto",
    weights: dict | None = None,
) -> Channels:
    weights = weights or W.get_weights()
    cfw = weights["channel_feature_weights"]
    baseline = weights["feature_baseline"]
    profiles = weights["content_type_profiles"]
    profile = profiles.get(content_type, profiles["auto"])

    out = {}
    for ch in CHANNELS:
        spec = cfw[ch]
        acc = np.full(grid.n, float(spec.get("_bias", 0.0)))
        for fname, w in spec.items():
            if fname == "_bias":
                continue
            acc = acc + w * _feature(features, fname, grid, baseline)
        gain = float(profile.get(ch, 1.0))
        out[ch] = clamp01(smooth(acc * gain, 3))

    return Channels(
        grid=grid,
        reward=out["reward"],
        attention=out["attention"],
        emotion=out["emotion"],
        aversion=out["aversion"],
        language=out["language"],
        content_type=content_type,
    )


def infer_content_type(features: Dict[str, np.ndarray], scalars: Dict) -> str:
    """Heuristic auto content-type classification.

    talking_head: sustained face presence + speech, low cut rate.
    edit: high cut rate / motion, sparse speech.
    vlog: moderate everything (default when mixed).
    """
    face = features.get("face_presence")
    cut = features.get("cut_rate")
    speech = scalars.get("word_count", 0) or scalars.get("n_words", 0)
    have_speech = scalars.get("have_speech", speech > 0)

    face_frac = float(np.mean(face)) if face is not None and len(face) else 0.0
    cut_mean = float(np.mean(cut)) if cut is not None and len(cut) else 0.0

    if face_frac > 0.45 and have_speech and cut_mean < 0.35:
        return "talking_head"
    if cut_mean > 0.45 and not have_speech:
        return "edit"
    if cut_mean > 0.5:
        return "edit"
    if have_speech:
        return "vlog"
    return "auto"
