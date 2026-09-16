"""Small numeric helpers shared across the pipeline.

Everything in NeuroViral is aligned to a common *time grid* sampled at a fixed
step (default 0.5s / 2 Hz). Extractors emit per-timestep feature arrays on this
grid so the brain layer can fuse them without resampling headaches.

Only depends on numpy so it is import-safe in every environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

DEFAULT_DT = 0.5  # seconds per timestep (2 Hz)


@dataclass(frozen=True)
class TimeGrid:
    """A uniform time grid covering [0, duration)."""

    duration: float
    dt: float = DEFAULT_DT

    @property
    def n(self) -> int:
        return max(1, int(math.ceil(self.duration / self.dt)))

    @property
    def times(self) -> np.ndarray:
        """Left-edge timestamp of each bin, in seconds."""
        return np.arange(self.n, dtype=float) * self.dt

    def index_of(self, t: float) -> int:
        return int(np.clip(int(t / self.dt), 0, self.n - 1))

    def zeros(self) -> np.ndarray:
        return np.zeros(self.n, dtype=float)

    def full(self, value: float) -> np.ndarray:
        return np.full(self.n, float(value), dtype=float)


def clamp01(x):
    """Clamp scalar or array into [0, 1]."""
    return np.clip(x, 0.0, 1.0)


def safe_norm(x: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Min-max normalize into [0,1]; robust to constant / empty arrays."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    if lo is None:
        lo = float(np.nanmin(x))
    if hi is None:
        hi = float(np.nanmax(x))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros_like(x)
    return clamp01((x - lo) / (hi - lo))


def smooth(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple centered moving-average smoothing; window in timesteps."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or window <= 1:
        return x
    window = min(window, x.size)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def resample_to_grid(
    values: Sequence[float],
    src_times: Sequence[float],
    grid: TimeGrid,
    fill: float = 0.0,
    agg: str = "mean",
) -> np.ndarray:
    """Bin arbitrary (time, value) samples onto the grid.

    agg: "mean" | "max" | "sum" per bin. Empty bins get `fill`.
    """
    out = grid.full(fill)
    values = np.asarray(values, dtype=float)
    src_times = np.asarray(src_times, dtype=float)
    if values.size == 0:
        return out
    counts = np.zeros(grid.n)
    acc = np.zeros(grid.n)
    started = np.zeros(grid.n, dtype=bool)
    for t, v in zip(src_times, values):
        if not math.isfinite(v):
            continue
        i = grid.index_of(float(t))
        if agg == "max":
            acc[i] = v if not started[i] else max(acc[i], v)
        else:  # mean or sum
            acc[i] += v
        counts[i] += 1
        started[i] = True
    for i in range(grid.n):
        if not started[i]:
            continue
        if agg == "mean":
            out[i] = acc[i] / max(1.0, counts[i])
        else:  # sum or max
            out[i] = acc[i]
    return out


def events_to_rate(event_times: Iterable[float], grid: TimeGrid) -> np.ndarray:
    """Count events falling in each bin, normalized to events-per-second."""
    out = grid.zeros()
    for t in event_times:
        i = grid.index_of(float(t))
        out[i] += 1.0
    return out / grid.dt


def integrate(x: np.ndarray, dt: float = DEFAULT_DT) -> float:
    return float(np.sum(np.asarray(x, dtype=float)) * dt)


def window_mean(x: np.ndarray, grid: TimeGrid, t0: float, t1: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    i0 = grid.index_of(t0)
    i1 = max(i0 + 1, grid.index_of(t1) + 1)
    seg = x[i0:i1]
    return float(np.mean(seg)) if seg.size else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class ExtractResult:
    """Standard return type for every extractor.

    `series` holds named per-timestep feature arrays (all length grid.n).
    `scalars` holds summary numbers. `degraded` flags fallback/neutral output,
    with human-readable `notes` explaining why.
    """

    series: dict = field(default_factory=dict)
    scalars: dict = field(default_factory=dict)
    degraded: bool = False
    notes: list = field(default_factory=list)

    def note(self, msg: str, degraded: bool = False) -> None:
        self.notes.append(msg)
        if degraded:
            self.degraded = True
