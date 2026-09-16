"""Unit tests for the brain layer on SYNTHETIC feature inputs (no models)."""
from __future__ import annotations

import numpy as np

from neuroviral.brain.channels import CHANNELS, build_channels, infer_content_type
from neuroviral.brain.dynamics import derive
from neuroviral.brain import weights as W
from neuroviral.utils import TimeGrid


def _grid(dur=10.0):
    return TimeGrid(duration=dur)


def test_channels_output_shape_and_range():
    grid = _grid()
    feats = {"motion": grid.full(0.8), "rms_energy": grid.full(0.7)}
    ch = build_channels(feats, grid)
    for c in CHANNELS:
        arr = getattr(ch, c)
        assert arr.shape[0] == grid.n
        assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_dead_air_raises_aversion():
    grid = _grid()
    calm = {"dead_air": grid.zeros(), "motion": grid.full(0.6)}
    loud_dead = {"dead_air": grid.full(1.0), "low_motion": grid.full(1.0),
                 "monotone": grid.full(1.0), "motion": grid.zeros()}
    ch_calm = build_channels(calm, grid)
    ch_dead = build_channels(loud_dead, grid)
    assert ch_dead.aversion.mean() > ch_calm.aversion.mean()


def test_curiosity_raises_reward():
    grid = _grid()
    base = build_channels({}, grid)
    hooked = build_channels(
        {"curiosity": grid.full(1.0), "hook_phrase": grid.full(1.0)}, grid)
    assert hooked.reward.mean() > base.reward.mean()


def test_dead_air_lowers_completion_and_hook():
    grid = _grid()
    good = build_channels(
        {"motion": grid.full(0.8), "curiosity": grid.full(0.6),
         "rms_energy": grid.full(0.7)}, grid)
    bad_feats = {
        "dead_air": grid.full(1.0), "slow_start": grid.full(1.0),
        "low_motion": grid.full(1.0), "monotone": grid.full(1.0),
        "motion": grid.zeros(),
    }
    bad = build_channels(bad_feats, grid)
    dg = derive(good)
    db = derive(bad)
    assert db.predicted_completion < dg.predicted_completion
    assert db.hook < dg.hook


def test_inserting_dead_air_monotonic():
    """Progressively more dead air => monotonically higher aversion, lower completion."""
    grid = _grid()
    prev_av, prev_comp = -1, 2.0
    for level in (0.0, 0.3, 0.6, 1.0):
        feats = {"dead_air": grid.full(level), "low_motion": grid.full(level),
                 "motion": grid.full(1.0 - level)}
        ch = build_channels(feats, grid)
        dyn = derive(ch)
        av = ch.aversion.mean()
        assert av >= prev_av - 1e-9
        assert dyn.predicted_completion <= prev_comp + 1e-9
        prev_av, prev_comp = av, dyn.predicted_completion


def test_content_type_profiles_differ():
    grid = _grid()
    feats = {"motion": grid.full(0.8), "cut_rate": grid.full(0.8),
             "self_reference": grid.full(0.8), "clarity": grid.full(0.8)}
    edit = build_channels(feats, grid, content_type="edit")
    talk = build_channels(feats, grid, content_type="talking_head")
    # edit should lean attention-heavy relative to language vs talking_head
    assert edit.attention.mean() > talk.attention.mean()
    assert talk.language.mean() > edit.language.mean()


def test_loopability_high_when_head_matches_tail():
    grid = _grid()
    # symmetric activation => head ~ tail => higher loop than a ramp
    flat = build_channels({"motion": grid.full(0.6), "rms_energy": grid.full(0.6)}, grid)
    ramp_feats = {"motion": np.linspace(0, 1, grid.n),
                  "rms_energy": np.linspace(0, 1, grid.n)}
    ramp = build_channels(ramp_feats, grid)
    assert derive(flat).loopability >= derive(ramp).loopability - 1e-6


def test_infer_content_type():
    grid = _grid()
    talk_feats = {"face_presence": grid.full(0.9), "cut_rate": grid.full(0.1)}
    assert infer_content_type(talk_feats, {"have_speech": True, "word_count": 20}) == "talking_head"
    edit_feats = {"face_presence": grid.zeros(), "cut_rate": grid.full(0.9)}
    assert infer_content_type(edit_feats, {"have_speech": False, "word_count": 0}) == "edit"


def test_weights_are_the_only_tunables():
    """Sanity: get_weights returns all required groups (calibration touches only this)."""
    w = W.get_weights()
    for key in ("channel_feature_weights", "content_type_profiles",
                "platform_weights", "dynamics", "subscores"):
        assert key in w
