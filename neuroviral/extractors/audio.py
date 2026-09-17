"""Audio features: RMS energy dynamics, energy build-up, prosody, onsets.

Energy features are computed with plain numpy so they work whenever an audio
track exists. Pitch/tempo prosody uses librosa when available and falls back to
a zero-crossing-rate proxy otherwise.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..io.media import AudioWaveform
from ..utils import ExtractResult, TimeGrid, resample_to_grid, safe_norm, smooth


def _frame_rms(samples: np.ndarray, sr: int, hop_s: float = 0.05, win_s: float = 0.1):
    hop = max(1, int(sr * hop_s))
    win = max(hop, int(sr * win_s))
    n = samples.size
    times = []
    rms = []
    for start in range(0, max(1, n - win + 1), hop):
        frame = samples[start:start + win]
        if frame.size == 0:
            continue
        rms.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-9))
        times.append((start + win / 2) / sr)
    return np.asarray(times), np.asarray(rms)


def _frame_zcr(samples: np.ndarray, sr: int, hop_s: float = 0.05, win_s: float = 0.1):
    hop = max(1, int(sr * hop_s))
    win = max(hop, int(sr * win_s))
    n = samples.size
    times, zcr = [], []
    for start in range(0, max(1, n - win + 1), hop):
        frame = samples[start:start + win]
        if frame.size < 2:
            continue
        crossings = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        zcr.append(float(crossings))
        times.append((start + win / 2) / sr)
    return np.asarray(times), np.asarray(zcr)


def extract(wave: Optional[AudioWaveform], grid: TimeGrid) -> ExtractResult:
    res = ExtractResult()
    if wave is None or wave.samples.size == 0:
        for name in ("rms_energy", "energy_buildup", "pitch_var", "monotone"):
            res.series[name] = grid.zeros()
        res.scalars["mean_energy"] = 0.0
        res.note("no audio track, audio features neutral/zero", degraded=True)
        return res

    samples = np.asarray(wave.samples, dtype=np.float32)
    sr = wave.sr

    t_rms, rms = _frame_rms(samples, sr)
    rms_series = safe_norm(resample_to_grid(rms, t_rms, grid, agg="mean"))
    rms_series = smooth(rms_series, 3)
    res.series["rms_energy"] = rms_series

    # Energy build-up = smoothed positive derivative of energy.
    deriv = np.diff(rms_series, prepend=rms_series[:1])
    res.series["energy_buildup"] = safe_norm(smooth(np.clip(deriv, 0, None), 3))

    # Prosody: pitch variance. librosa if present, else ZCR-variance proxy.
    pitch_series = _pitch_variance(samples, sr, grid, res)
    res.series["pitch_var"] = pitch_series

    # Monotone = low local variability of both energy and pitch.
    energy_var = _local_variability(rms_series)
    monotone = np.clip(1.0 - 0.5 * (energy_var + pitch_series), 0, 1)
    # Only meaningful where there is sound; scale by loudness presence.
    monotone = monotone * (rms_series > 0.08)
    res.series["monotone"] = smooth(monotone, 3)

    res.scalars["mean_energy"] = float(np.mean(rms_series))
    return res


def _local_variability(x: np.ndarray, w: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(x.size):
        lo = max(0, i - w // 2)
        hi = min(x.size, i + w // 2 + 1)
        out[i] = np.std(x[lo:hi])
    return safe_norm(out)


def _pitch_variance(samples: np.ndarray, sr: int, grid: TimeGrid, res: ExtractResult) -> np.ndarray:
    try:
        import librosa  # type: ignore

        f0, _, _ = librosa.pyin(
            samples.astype(np.float32),
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
        hop = 512
        times = librosa.times_like(f0, sr=sr, hop_length=hop)
        f0 = np.where(np.isfinite(f0), f0, np.nan)
        # local std of pitch over ~1s windows
        var = np.zeros_like(f0)
        wlen = max(1, int(1.0 * sr / hop))
        for i in range(f0.size):
            lo = max(0, i - wlen // 2)
            hi = min(f0.size, i + wlen // 2 + 1)
            seg = f0[lo:hi]
            seg = seg[np.isfinite(seg)]
            var[i] = np.std(seg) if seg.size > 1 else 0.0
        series = safe_norm(resample_to_grid(var, times, grid, agg="mean"))
        return smooth(series, 3)
    except Exception:
        res.note("librosa unavailable, pitch via zero-crossing proxy", degraded=False)
        t_zcr, zcr = _frame_zcr(samples, sr)
        zseries = resample_to_grid(zcr, t_zcr, grid, agg="mean")
        return smooth(_local_variability(safe_norm(zseries)), 3)
