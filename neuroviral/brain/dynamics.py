"""Derived temporal dynamics that scores read from.

* Hook strength (0-3s): integrated Reward + Attention minus Aversion.
* Retention curve: a per-second survival model; Aversion accelerates drop-off,
  Reward/novelty recovers it, salience holds attention. Integrated -> predicted
  completion %.
* Loopability: similarity of the last ~1s to the first ~1s (channel-space +
  optional frame/audio), plus whether an open loop resolves at the very end.
* Peak-end: emotional arousal at the peak and at the ending (peak-end rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..utils import TimeGrid, clamp01, cosine, window_mean
from .channels import Channels
from . import weights as W


@dataclass
class Dynamics:
    hook: float                      # 0-1
    retention_curve: np.ndarray      # survival probability per timestep, 0-1
    predicted_completion: float      # 0-1
    loopability: float               # 0-1
    peak_end: float                  # 0-1
    arousal: float                   # 0-1
    arousal_peak_time: float
    visual_hook: float = 0.0         # 0-1, first hook_window_s
    text_hook: float = 0.0           # 0-1, first hook_window_s (neutral-low if no speech)
    audio_hook: float = 0.0          # 0-1, first hook_window_s
    uniqueness: float = 0.0          # 0-1, semantic + visual content novelty
    storytelling: float = 0.0        # 0-1, narrative-arc quality
    dropoff_zones: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _feat(features: dict, grid: TimeGrid, baseline: dict, name: str) -> np.ndarray:
    """Pull a feature series with baseline fallback (mirrors channels._feature)."""
    arr = features.get(name)
    if arr is None:
        return grid.full(float(baseline.get(name, 0.0)))
    return np.asarray(arr, dtype=float)


def _visual_hook(features: dict, grid: TimeGrid, params: dict, baseline: dict) -> float:
    win = params["hook_window_s"]
    def w(name):
        return window_mean(_feat(features, grid, baseline, name), grid, 0, win)
    raw = (0.28 * w("motion") + 0.22 * w("cut_rate") + 0.18 * w("face_size")
           + 0.10 * w("face_presence") + 0.12 * w("color_contrast")
           + 0.10 * w("text_onscreen"))
    raw -= 0.25 * w("low_motion")           # dead visual = penalty
    return float(clamp01(raw * params.get("subhook_gain", 1.15)))


def _text_hook(features: dict, grid: TimeGrid, params: dict, baseline: dict,
               have_speech: bool) -> float:
    if not have_speech:
        return 0.15                          # neutral-low: no script to score, not a failure
    win = params["hook_window_s"]
    def w(name):
        return window_mean(_feat(features, grid, baseline, name), grid, 0, win)
    raw = (0.32 * w("curiosity") + 0.30 * w("hook_phrase") + 0.16 * w("novelty")
           + 0.12 * w("self_reference") + 0.10 * w("sentiment_arousal"))
    return float(clamp01(raw * params.get("subhook_gain", 1.15)))


def _audio_hook(features: dict, grid: TimeGrid, params: dict, baseline: dict) -> float:
    win = params["hook_window_s"]
    def w(name):
        return window_mean(_feat(features, grid, baseline, name), grid, 0, win)
    raw = (0.34 * w("rms_energy") + 0.24 * w("energy_buildup")
           + 0.22 * w("pitch_var") + 0.20 * w("speech_rate"))
    raw -= 0.30 * w("dead_air") + 0.25 * w("slow_start")   # dead-air / slow start = penalty
    return float(clamp01(raw * params.get("subhook_gain", 1.15)))


def _uniqueness(ch: Channels, features: dict) -> float:
    """Semantic novelty (primary) + visual variety (secondary).

    text_semantics emits a *flat* 0.3 novelty array when there is no transcript;
    that synthetic baseline is detected and routed to a separate capped branch so
    it can't masquerade as genuinely 'average' content.
    """
    grid = ch.grid
    nov = _feat(features, grid, {"novelty": 0.3}, "novelty")
    cut = _feat(features, grid, {"cut_rate": 0.2}, "cut_rate")
    mot = _feat(features, grid, {"motion": 0.3}, "motion")

    is_baseline = (nov.size == 0) or (float(np.ptp(nov)) < 1e-3
                                      and abs(float(np.mean(nov)) - 0.3) < 1e-3)
    visual_variety = clamp01(0.5 * float(np.std(cut)) * 2.5
                             + 0.5 * float(np.std(mot)) * 2.5)
    if is_baseline:
        # No real semantic signal, so score on visual variety only, capped ~0.60.
        return float(clamp01(0.15 + 0.45 * visual_variety))

    nov_mean = float(np.mean(nov))
    nov_var = float(np.std(nov))
    semantic = clamp01(0.7 * nov_mean + 0.9 * nov_var)   # mean bounded, variety rewarded
    return float(clamp01(0.65 * semantic + 0.35 * visual_variety))


def _storytelling(ch: Channels, features: dict, params: dict) -> float:
    """Narrative arc: reward builds, emotion varies, and the ending resolves."""
    grid = ch.grid
    n = grid.n
    if n < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, n)
    slope = float(np.polyfit(x, ch.reward, 1)[0])       # reward rise over the clip
    build = clamp01(0.5 + slope)

    emo_range = float(np.percentile(ch.emotion, 90) - np.percentile(ch.emotion, 10))
    variation = clamp01(emo_range * 2.0)

    payoff = _feat(features, grid, {"payoff": 0.0}, "payoff")
    end_payoff = window_mean(payoff, grid, grid.duration * (2 / 3), grid.duration)
    end_reward = window_mean(ch.reward, grid, grid.duration - params["end_window_s"],
                             grid.duration)
    resolution = clamp01(0.5 * clamp01(end_payoff * 3.0) + 0.5 * end_reward)

    monotone_pen = 1.0 if emo_range >= 0.08 else emo_range / 0.08
    arc = 0.30 * build + 0.25 * variation + 0.45 * resolution
    return float(clamp01(arc * monotone_pen))


def _hook(ch: Channels, params: dict, sub: dict) -> float:
    grid = ch.grid
    win = params["hook_window_s"]
    reward = window_mean(ch.reward, grid, 0, win)
    attention = window_mean(ch.attention, grid, 0, win)
    aversion = window_mean(ch.aversion, grid, 0, win)
    raw = (sub["hook_reward_w"] * reward
           + sub["hook_attention_w"] * attention
           - sub["hook_aversion_w"] * aversion)
    return float(clamp01(raw * sub["hook_gain"]))


def _retention(ch: Channels, params: dict):
    grid = ch.grid
    dt = grid.dt
    base = params["base_dropoff_per_s"]
    av_g = params["aversion_dropoff_gain"]
    rw_g = params["reward_recovery_gain"]
    at_g = params["attention_hold_gain"]

    surv = np.ones(grid.n)
    p = 1.0
    novelty_recovery = np.clip(np.diff(ch.reward, prepend=ch.reward[:1]), 0, None)
    for i in range(grid.n):
        hazard = base
        hazard += av_g * ch.aversion[i]
        hazard -= rw_g * (0.5 * ch.reward[i] + 0.5 * novelty_recovery[i])
        hazard -= at_g * ch.attention[i]
        hazard = max(0.0, hazard)
        p = p * float(np.exp(-hazard * dt))
        surv[i] = p
    completion = float(np.mean(surv))
    return surv, completion


def _dropoff_zones(ch: Channels, surv: np.ndarray, thresh: float = 0.5) -> List[dict]:
    grid = ch.grid
    zones = []
    in_zone = False
    start = 0.0
    for i in range(grid.n):
        risky = ch.aversion[i] >= thresh
        if risky and not in_zone:
            in_zone = True
            start = grid.times[i]
        elif not risky and in_zone:
            in_zone = False
            zones.append({"start": round(start, 2),
                          "end": round(grid.times[i], 2),
                          "severity": "high"})
    if in_zone:
        zones.append({"start": round(start, 2),
                      "end": round(grid.duration, 2), "severity": "high"})
    return zones


def _loopability(ch: Channels, params: dict,
                 first_frame=None, last_frame=None) -> float:
    grid = ch.grid
    tail = params["loop_tail_s"]
    head_idx = slice(0, max(1, grid.index_of(tail) + 1))
    tail_idx = slice(max(0, grid.n - (grid.index_of(tail) + 1)), grid.n)

    mat = np.stack([ch.reward, ch.attention, ch.emotion, ch.language], axis=0)
    head_vec = mat[:, head_idx].mean(axis=1)
    tail_vec = mat[:, tail_idx].mean(axis=1)
    channel_sim = 0.5 * (cosine(head_vec, tail_vec) + 1.0)  # map [-1,1]->[0,1]

    visual_sim = channel_sim
    if first_frame is not None and last_frame is not None:
        try:
            a = np.asarray(first_frame, dtype=float).ravel()
            b = np.asarray(last_frame, dtype=float).ravel()
            m = min(a.size, b.size)
            visual_sim = 0.5 * (cosine(a[:m], b[:m]) + 1.0)
        except Exception:
            pass

    vw = params["loop_visual_weight"]
    aw = params["loop_audio_weight"]
    loop = vw * visual_sim + aw * channel_sim
    # A held reward/curiosity at the end (unresolved loop pulling replay) helps.
    end_reward = window_mean(ch.reward, grid, grid.duration - tail, grid.duration)
    loop = 0.8 * loop + 0.2 * end_reward
    return float(clamp01(loop))


def _peak_end(ch: Channels, params: dict) -> tuple:
    grid = ch.grid
    arousal = ch.emotion
    peak_val = float(np.max(arousal)) if arousal.size else 0.0
    peak_time = float(grid.times[int(np.argmax(arousal))]) if arousal.size else 0.0
    end_val = window_mean(arousal, grid, grid.duration - params["end_window_s"], grid.duration)
    pe = (params["peak_end_peak_weight"] * peak_val
          + params["peak_end_end_weight"] * end_val)
    return float(clamp01(pe)), peak_val, peak_time


def derive(
    ch: Channels,
    weights: dict | None = None,
    first_frame=None,
    last_frame=None,
    features: dict | None = None,
    have_speech: bool = True,
) -> Dynamics:
    weights = weights or W.get_weights()
    params = weights["dynamics"]
    sub = weights["subscores"]
    baseline = weights["feature_baseline"]
    features = features or {}

    hook = _hook(ch, params, sub)
    surv, completion = _retention(ch, params)
    zones = _dropoff_zones(ch, surv)
    loop = _loopability(ch, params, first_frame, last_frame)
    peak_end, peak_val, peak_time = _peak_end(ch, params)

    # Hook breakdown (diagnostic; overall `hook` above is unchanged for scoring).
    visual_hook = _visual_hook(features, ch.grid, params, baseline)
    text_hook = _text_hook(features, ch.grid, params, baseline, have_speech)
    audio_hook = _audio_hook(features, ch.grid, params, baseline)

    uniqueness = _uniqueness(ch, features)
    storytelling = _storytelling(ch, features, params)

    # Arousal sub-score = high-percentile emotion activation.
    pct = sub["arousal_percentile"]
    arousal = float(np.percentile(ch.emotion, pct)) if ch.emotion.size else 0.0

    return Dynamics(
        hook=hook,
        retention_curve=surv,
        predicted_completion=completion,
        loopability=loop,
        peak_end=peak_end,
        arousal=arousal,
        arousal_peak_time=peak_time,
        visual_hook=visual_hook,
        text_hook=text_hook,
        audio_hook=audio_hook,
        uniqueness=uniqueness,
        storytelling=storytelling,
        dropoff_zones=zones,
    )
