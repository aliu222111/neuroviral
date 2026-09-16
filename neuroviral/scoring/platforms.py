"""Channel dynamics -> TikTok / Reels virality scores with confidence band.

Sub-scores (all 0-1): hook, completion, loopability, peak_end, arousal.
Each platform weights them differently (``brain/weights.PLATFORM_WEIGHTS``).
The composite is mapped to 0-100 with a mild concave curve (gamma<1, which
lifts mid-range scores). The confidence band
reflects (a) whether calibration data has been fitted and (b) how many
extractors degraded to neutral output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..brain.dynamics import Dynamics
from ..brain import weights as W


@dataclass
class PlatformScore:
    platform: str
    score: float                 # 0-100
    confidence: float            # 0-1
    low: float                   # 0-100 band low
    high: float                  # 0-100 band high
    subscores: Dict[str, float] = field(default_factory=dict)


def _subscores(dyn: Dynamics) -> Dict[str, float]:
    return {
        "hook": float(dyn.hook),
        "completion": float(dyn.predicted_completion),
        "loopability": float(dyn.loopability),
        "peak_end": float(dyn.peak_end),
        "arousal": float(dyn.arousal),
    }


def _composite(subs: Dict[str, float], platform_w: Dict[str, float]) -> float:
    total_w = sum(platform_w.values()) or 1.0
    return sum(subs[k] * platform_w.get(k, 0.0) for k in subs) / total_w


def _to_100(composite: float, gamma: float) -> float:
    composite = max(0.0, min(1.0, composite))
    return round(100.0 * (composite ** gamma), 1)


def _confidence(n_degraded: int, calibrated: bool, sub: dict) -> float:
    base = sub["confidence_base"]
    if calibrated:
        base = min(0.9, base + 0.2)
    conf = base - n_degraded * sub["confidence_degraded_penalty"]
    return float(max(0.25, min(0.95, conf)))


def score_platform(
    dyn: Dynamics,
    platform: str,
    n_degraded: int = 0,
    weights: dict | None = None,
) -> PlatformScore:
    weights = weights or W.get_weights()
    sub_params = weights["subscores"]
    platform_w = weights["platform_weights"][platform]
    subs = _subscores(dyn)
    composite = _composite(subs, platform_w)
    score = _to_100(composite, sub_params["score_curve_gamma"])
    conf = _confidence(n_degraded, weights.get("_calibrated", False), sub_params)
    # Band widens as confidence drops.
    half = round((1.0 - conf) * 22.0, 1)
    return PlatformScore(
        platform=platform,
        score=score,
        confidence=round(conf, 2),
        low=round(max(0.0, score - half), 1),
        high=round(min(100.0, score + half), 1),
        subscores={k: round(v, 3) for k, v in subs.items()},
    )


def score_all(
    dyn: Dynamics,
    platforms: List[str],
    n_degraded: int = 0,
    weights: dict | None = None,
) -> Dict[str, PlatformScore]:
    weights = weights or W.get_weights()
    return {p: score_platform(dyn, p, n_degraded, weights) for p in platforms}
