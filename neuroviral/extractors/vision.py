"""Vision features: cut detection, motion energy, color contrast/saturation.

Motion and color are computed with numpy directly from sampled frames, so they
work with no extra deps. Shot/cut detection uses PySceneDetect/OpenCV when
present and otherwise falls back to a frame-difference peak detector.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..utils import ExtractResult, TimeGrid, events_to_rate, resample_to_grid, safe_norm, smooth


def _luma(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _saturation(frame: np.ndarray) -> float:
    f = frame.astype(np.float32) / 255.0
    mx = f.max(axis=-1)
    mn = f.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    return float(np.mean(sat))


def extract(frames: List[np.ndarray], frame_fps: float, grid: TimeGrid) -> ExtractResult:
    res = ExtractResult()
    if not frames or frame_fps <= 0:
        for name in ("motion", "cut_rate", "color_contrast", "color_saturation",
                     "low_motion", "long_shot", "text_onscreen"):
            res.series[name] = grid.full(0.0)
        res.series["color_contrast"] = grid.full(0.4)
        res.series["color_saturation"] = grid.full(0.4)
        res.note("no video frames, vision features neutral", degraded=True)
        return res

    times = np.arange(len(frames)) / frame_fps
    lumas = [_luma(f) for f in frames]

    # Motion energy: mean abs luma diff between consecutive frames.
    motion_vals = [0.0]
    for i in range(1, len(lumas)):
        d = np.mean(np.abs(lumas[i] - lumas[i - 1])) / 255.0
        motion_vals.append(float(d))
    motion_vals = np.asarray(motion_vals)
    motion = safe_norm(resample_to_grid(motion_vals, times, grid, agg="mean"))
    motion = smooth(motion, 3)
    res.series["motion"] = motion
    res.series["low_motion"] = smooth(np.clip(0.6 - motion, 0, 1) / 0.6, 3)

    # Color stats.
    contrast_vals = np.asarray([float(np.std(l) / 128.0) for l in lumas])
    sat_vals = np.asarray([_saturation(f) for f in frames])
    res.series["color_contrast"] = safe_norm(resample_to_grid(contrast_vals, times, grid, agg="mean"))
    res.series["color_saturation"] = resample_to_grid(np.clip(sat_vals, 0, 1), times, grid, agg="mean")

    # Cuts.
    cut_times = _detect_cuts(frames, times, motion_vals, res)
    res.series["cut_rate"] = safe_norm(smooth(events_to_rate(cut_times, grid), 3))

    # Long-shot: time since last cut (normalized). Big = held too long.
    long_shot = _time_since_last_cut(cut_times, grid)
    res.series["long_shot"] = long_shot

    # Rough on-screen-text proxy: fraction of very-bright high-contrast pixels.
    text_vals = []
    for l in lumas:
        bright = l > 220
        text_vals.append(float(np.mean(bright)))
    res.series["text_onscreen"] = safe_norm(resample_to_grid(np.asarray(text_vals), times, grid, agg="mean"))

    res.scalars["n_cuts"] = len(cut_times)
    res.scalars["mean_motion"] = float(np.mean(motion))
    res._first_frame = frames[0]  # type: ignore[attr-defined]
    res._last_frame = frames[-1]  # type: ignore[attr-defined]
    return res


def _detect_cuts(frames, times, motion_vals, res: ExtractResult) -> List[float]:
    try:
        from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore
        # PySceneDetect needs a file; we already have decoded frames, so use the
        # lightweight numpy detector to avoid re-decoding. Fall through.
        raise ImportError
    except Exception:
        pass
    # Frame-difference cut detector (PySceneDetect-style). A hard cut is a rising
    # local peak in content-change that clears an absolute floor and stands out
    # from the local baseline, which is robust to busy footage (where a global
    # median+std threshold mis-handles, since one big jump inflates std above
    # itself and suppresses detection).
    m = np.asarray(motion_vals)
    if m.size < 2:
        return []
    floor = 0.12  # normalized mean-abs luma diff; hard cuts jump well past this
    cuts = []
    for i in range(1, m.size):
        rising = m[i] > m[i - 1]
        peak = (i == m.size - 1) or (m[i] >= m[i + 1])
        baseline = float(np.median(m[max(0, i - 6):i])) if i > 0 else 0.0
        if m[i] >= floor and m[i] >= baseline * 1.6 and rising and peak:
            if not cuts or times[i] - cuts[-1] > 0.3:
                cuts.append(float(times[i]))
    return cuts


def _time_since_last_cut(cut_times: List[float], grid: TimeGrid) -> np.ndarray:
    out = grid.zeros()
    cuts = sorted(cut_times)
    last = 0.0
    ci = 0
    for i, t in enumerate(grid.times):
        while ci < len(cuts) and cuts[ci] <= t:
            last = cuts[ci]
            ci += 1
        out[i] = t - last
    # Normalize: 6s+ held shot => 1.0
    return np.clip(out / 6.0, 0, 1)
