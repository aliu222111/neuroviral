"""Central home for EVERY tunable coefficient in NeuroViral.

Design-for-calibration principle: the calibration re-fit (``calibration/fit.py``)
only ever writes here (via :func:`load_overrides`) — it never touches pipeline
code. Research-derived defaults live in this file; a JSON override file produced
by calibration is layered on top when present.

Weight groups
-------------
* ``CHANNEL_FEATURE_WEIGHTS`` — linear map from extracted features to each of the
  five neural channels (plus a bias). This encodes the mapping table in the spec.
* ``CONTENT_TYPE_PROFILES`` — per-content-type channel gain multipliers
  (talking_head / edit / vlog / auto).
* ``PLATFORM_WEIGHTS`` — how TikTok vs Reels weight the derived sub-scores.
* ``DYNAMICS`` — decay / hook-window / loop parameters.
* ``SUBSCORES`` — misc thresholds used when translating channels to sub-scores.

All feature series are 0–1 aligned to the common time grid. A missing feature is
treated as its neutral baseline (see ``FEATURE_BASELINE``) so channels always
compute even under graceful degradation.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Feature -> channel linear weights (the mapping table, as coefficients).
# Positive weight => feature raises the channel; negative => lowers it.
# Each channel is: sigmoid-free clamped( bias + sum(w_i * feature_i) ).
# ---------------------------------------------------------------------------
CHANNEL_FEATURE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Nucleus accumbens: reward / anticipation. Top forecaster in the research.
    "reward": {
        "_bias": 0.12,
        "curiosity": 0.55,      # open loops / curiosity-gap phrasing
        "hook_phrase": 0.35,    # "you won't believe..." style
        "novelty": 0.25,        # semantic novelty / surprise
        "energy_buildup": 0.30, # rising audio energy = build-up
        "payoff": 0.20,         # detected resolution/payoff moments
    },
    # Salience / attention network: drives the first-3s hold.
    "attention": {
        "_bias": 0.10,
        "motion": 0.35,
        "cut_rate": 0.30,
        "face_size": 0.25,
        "face_presence": 0.10,
        "color_contrast": 0.15,
        "color_saturation": 0.10,
        "text_onscreen": 0.15,
    },
    # Amygdala / insula: emotional arousal. Predicts sharing.
    "emotion": {
        "_bias": 0.10,
        "rms_energy": 0.30,
        "pitch_var": 0.25,
        "speech_rate": 0.15,
        "expression_intensity": 0.25,
        "sentiment_arousal": 0.25,
    },
    # Anterior insula (aversion / drop-off). This is the "scroll trigger".
    "aversion": {
        "_bias": 0.05,
        "dead_air": 0.55,
        "low_motion": 0.30,
        "monotone": 0.30,
        "long_shot": 0.20,
        "slow_start": 0.35,
    },
    # MPFC / language areas: self-relevance & message clarity -> sharing intent.
    "language": {
        "_bias": 0.10,
        "self_reference": 0.35,
        "novelty": 0.25,
        "cta": 0.25,
        "clarity": 0.30,
    },
}

# Neutral baseline for any feature an extractor could not produce.
FEATURE_BASELINE: Dict[str, float] = {
    "curiosity": 0.0, "hook_phrase": 0.0, "novelty": 0.3, "energy_buildup": 0.0,
    "payoff": 0.0, "motion": 0.3, "cut_rate": 0.2, "face_size": 0.0,
    "face_presence": 0.0, "color_contrast": 0.4, "color_saturation": 0.4,
    "text_onscreen": 0.0, "rms_energy": 0.3, "pitch_var": 0.3, "speech_rate": 0.3,
    "expression_intensity": 0.3, "sentiment_arousal": 0.3, "dead_air": 0.0,
    "low_motion": 0.0, "monotone": 0.0, "long_shot": 0.0, "slow_start": 0.0,
    "self_reference": 0.0, "cta": 0.0, "clarity": 0.5,
}

# ---------------------------------------------------------------------------
# Content-type channel gain multipliers. `auto` is the neutral default; the CLI
# can infer a type and swap the profile.
# ---------------------------------------------------------------------------
CONTENT_TYPE_PROFILES: Dict[str, Dict[str, float]] = {
    "auto": {
        "reward": 1.0, "attention": 1.0, "emotion": 1.0,
        "aversion": 1.0, "language": 1.0,
    },
    # Talking-head: language/message + emotion carry the video; dead air hurts most.
    "talking_head": {
        "reward": 0.95, "attention": 0.85, "emotion": 1.10,
        "aversion": 1.20, "language": 1.25,
    },
    # Edits/montages: salience & attention dominate; language matters less.
    "edit": {
        "reward": 1.05, "attention": 1.30, "emotion": 1.10,
        "aversion": 0.90, "language": 0.75,
    },
    # Vlogs: balanced, slight emotion/language lean, mild aversion tolerance.
    "vlog": {
        "reward": 1.0, "attention": 1.0, "emotion": 1.10,
        "aversion": 0.95, "language": 1.05,
    },
}

# ---------------------------------------------------------------------------
# Platform sub-score weighting. Sub-scores are all 0-1 before weighting.
# Weights per platform are normalized internally, so relative magnitude matters.
# TikTok = watch-time / hook / loop dominant. Reels = discovery, shareability,
# and slightly more forgiving of a marginally slower hook.
# ---------------------------------------------------------------------------
PLATFORM_WEIGHTS: Dict[str, Dict[str, float]] = {
    "tiktok": {
        "hook": 0.34,
        "completion": 0.26,
        "loopability": 0.18,
        "peak_end": 0.12,
        "arousal": 0.10,
    },
    "reels": {
        "hook": 0.26,
        "completion": 0.24,
        "loopability": 0.12,
        "peak_end": 0.20,
        "arousal": 0.18,
    },
}

# ---------------------------------------------------------------------------
# Derived-dynamics parameters.
# ---------------------------------------------------------------------------
DYNAMICS = {
    "hook_window_s": 3.0,        # first-N-seconds hook window
    "subhook_gain": 1.15,        # gain for the visual/text/audio hook breakdown
    "base_dropoff_per_s": 0.010, # baseline hazard of scrolling away, per second
    "aversion_dropoff_gain": 0.14,   # extra hazard per unit aversion
    "reward_recovery_gain": 0.06,    # reward/novelty recovers watch probability
    "attention_hold_gain": 0.03,     # salience reduces hazard
    "loop_tail_s": 1.0,          # window compared head-vs-tail for loopability
    "loop_visual_weight": 0.6,
    "loop_audio_weight": 0.4,
    "peak_end_peak_weight": 0.55,
    "peak_end_end_weight": 0.45,
    "end_window_s": 2.0,
}

# ---------------------------------------------------------------------------
# Sub-score translation thresholds / gains.
# ---------------------------------------------------------------------------
SUBSCORES = {
    "hook_reward_w": 0.9,
    "hook_attention_w": 1.0,
    "hook_aversion_w": 1.1,
    "hook_gain": 1.15,
    "arousal_percentile": 75,   # arousal sub-score = high-percentile emotion
    "confidence_base": 0.62,    # baseline confidence with no calibration data
    "confidence_degraded_penalty": 0.06,  # per degraded extractor
    "score_curve_gamma": 0.85,  # maps 0-1 composite to 0-100 (slightly convex)
}

# Overall score = weighted sub-scores, then scaled 0-100.
_OVERRIDE_ENV = "NEUROVIRAL_WEIGHTS"


def _default_override_path() -> Path:
    return Path(__file__).with_name("weights_calibrated.json")


def load_overrides() -> dict:
    """Load a calibration override JSON if one exists.

    Search order: ``$NEUROVIRAL_WEIGHTS`` env var, then
    ``brain/weights_calibrated.json``. Returns ``{}`` when absent (the no-op
    default until real post-analytics have been fitted).
    """
    candidates = []
    env = os.environ.get(_OVERRIDE_ENV)
    if env:
        candidates.append(Path(env))
    candidates.append(_default_override_path())
    for path in candidates:
        try:
            if path and path.exists():
                return json.loads(path.read_text())
        except Exception:
            continue
    return {}


def get_weights() -> dict:
    """Return the active weight bundle (defaults deep-merged with overrides)."""
    bundle = {
        "channel_feature_weights": deepcopy(CHANNEL_FEATURE_WEIGHTS),
        "feature_baseline": deepcopy(FEATURE_BASELINE),
        "content_type_profiles": deepcopy(CONTENT_TYPE_PROFILES),
        "platform_weights": deepcopy(PLATFORM_WEIGHTS),
        "dynamics": deepcopy(DYNAMICS),
        "subscores": deepcopy(SUBSCORES),
        "_calibrated": False,
    }
    overrides = load_overrides()
    if overrides:
        _deep_update(bundle, overrides)
        bundle["_calibrated"] = True
    return bundle


def _deep_update(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def save_overrides(overrides: dict, path: str | None = None) -> str:
    target = Path(path) if path else _default_override_path()
    target.write_text(json.dumps(overrides, indent=2))
    return str(target)


# The set of all feature names the brain layer understands.
ALL_FEATURES = tuple(sorted(FEATURE_BASELINE.keys()))
