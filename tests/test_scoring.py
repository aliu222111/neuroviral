"""Unit tests for scoring + recommendations (synthetic dynamics, no models)."""
from __future__ import annotations

import numpy as np

from neuroviral.brain.channels import build_channels
from neuroviral.brain.dynamics import derive
from neuroviral.scoring.platforms import score_all, score_platform
from neuroviral.scoring.recommendations import generate
from neuroviral.utils import TimeGrid


def _grid(dur=10.0):
    return TimeGrid(duration=dur)


def _good_channels(grid):
    return build_channels(
        {"motion": grid.full(0.8), "curiosity": grid.full(0.7),
         "hook_phrase": grid.full(0.8), "rms_energy": grid.full(0.7),
         "sentiment_arousal": grid.full(0.7), "self_reference": grid.full(0.6),
         "cut_rate": grid.full(0.6)}, grid)


def _bad_channels(grid):
    return build_channels(
        {"dead_air": grid.full(0.9), "slow_start": grid.full(1.0),
         "low_motion": grid.full(0.9), "monotone": grid.full(0.9),
         "motion": grid.zeros()}, grid)


def test_score_range_and_band():
    grid = _grid()
    dyn = derive(_good_channels(grid))
    s = score_platform(dyn, "tiktok")
    assert 0 <= s.score <= 100
    assert s.low <= s.score <= s.high
    assert 0 < s.confidence <= 1


def test_good_beats_bad_on_both_platforms():
    grid = _grid()
    good = derive(_good_channels(grid))
    bad = derive(_bad_channels(grid))
    for p in ("tiktok", "reels"):
        assert score_platform(good, p).score > score_platform(bad, p).score


def test_degraded_lowers_confidence():
    grid = _grid()
    dyn = derive(_good_channels(grid))
    clean = score_platform(dyn, "tiktok", n_degraded=0)
    degraded = score_platform(dyn, "tiktok", n_degraded=3)
    assert degraded.confidence < clean.confidence
    assert (degraded.high - degraded.low) >= (clean.high - clean.low)


def test_tiktok_weights_hook_more_than_reels():
    """A strong-hook / weak-shareability clip should favor TikTok."""
    grid = _grid()
    ch = build_channels(
        {"motion": grid.full(0.9), "curiosity": grid.full(0.9),
         "hook_phrase": grid.full(0.9), "cut_rate": grid.full(0.8)}, grid)
    dyn = derive(ch)
    scores = score_all(dyn, ["tiktok", "reels"])
    # hook is high, arousal/peak-end modest => tiktok should not trail reels
    assert scores["tiktok"].score >= scores["reels"].score - 1e-6


def test_recommendations_are_timestamped_and_ranked():
    grid = _grid()
    ch = _bad_channels(grid)
    dyn = derive(ch)
    recs = generate(ch, dyn, {})
    assert len(recs) >= 1
    lifts = [r.est_lift for r in recs]
    assert lifts == sorted(lifts, reverse=True)  # ranked by lift
    # at least one dead-air/slow-start rec should carry a timestamp
    assert any(r.t_start is not None for r in recs)


def test_recommendations_flag_dead_air_zone():
    grid = _grid()
    # dead air from 4s to 6s
    i0, i1 = grid.index_of(4.0), grid.index_of(6.0)
    feats = {"dead_air": grid.zeros(), "motion": grid.full(0.7)}
    feats["dead_air"][i0:i1] = 1.0
    ch = build_channels(feats, grid)
    dyn = derive(ch)
    recs = generate(ch, dyn, {})
    titles = " ".join(r.title.lower() for r in recs)
    assert "drop-off" in titles or "slow" in titles or "tighten" in titles
