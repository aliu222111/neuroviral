"""End-to-end pipeline tests on synthetic ffmpeg-generated clips.

Asserts the spec's face-validity direction:
  * high-energy fast-cut clip scores higher on Hook and TikTok than a
    slow/dead-air clip;
  * inserting dead air raises Aversion and lowers predicted completion;
  * the pipeline survives a very short, silent clip without crashing;
  * every run emits a calibration record.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from neuroviral.pipeline import analyze
from .conftest import has_ffmpeg


@has_ffmpeg
def test_high_energy_beats_slow_start(clips):
    good = analyze(clips["high"], platforms=["tiktok", "reels"])
    slow = analyze(clips["slow"], platforms=["tiktok", "reels"])

    # Hook: strong-hook clip must beat the slow start.
    assert good.dynamics.hook > slow.dynamics.hook
    # TikTok score meaningfully higher.
    assert good.scores["tiktok"].score > slow.scores["tiktok"].score

    # Dead-air slow clip: higher aversion, lower predicted completion.
    assert slow.channels.aversion.mean() > good.channels.aversion.mean()
    assert slow.dynamics.predicted_completion < good.dynamics.predicted_completion


@has_ffmpeg
def test_slow_clip_flags_slow_start_recommendation(clips):
    slow = analyze(clips["slow"])
    titles = " ".join(r.title.lower() for r in slow.recommendations)
    assert ("slow" in titles) or ("drop-off" in titles) or ("hook" in titles)


@has_ffmpeg
def test_short_silent_clip_does_not_crash(clips):
    res = analyze(clips["short"])
    assert 0 <= res.scores["tiktok"].score <= 100
    # No audio -> at least the audio/speech extractors degrade.
    assert res.n_degraded >= 1


@has_ffmpeg
def test_result_serializes_and_emits_calibration_record(clips):
    res = analyze(clips["high"])
    d = res.to_dict()
    # round-trips through JSON
    json.dumps(d)
    assert "calibration_record" in d
    rec = d["calibration_record"]
    assert rec["features_hash"]
    assert set(rec["channel_means"].keys()) == {
        "reward", "attention", "emotion", "aversion", "language"}
    assert "disclaimer" in d and "SIMULATION" in d["disclaimer"].upper()


@has_ffmpeg
def test_json_snapshot_shape(clips):
    res = analyze(clips["high"], platforms=["tiktok", "reels"])
    d = res.to_dict()
    assert set(d["scores"].keys()) == {"tiktok", "reels"}
    assert len(d["channels"]["reward"]) == len(d["times"])
    assert len(d["retention_curve"]) == len(d["times"])
    for p in ("tiktok", "reels"):
        s = d["scores"][p]
        assert set(s["subscores"].keys()) == {
            "hook", "completion", "loopability", "peak_end", "arousal"}


def test_calibration_fit_is_noop_without_data(tmp_path):
    from neuroviral.calibration.fit import fit
    empty = tmp_path / "none.jsonl"
    empty.write_text("")
    assert fit(str(empty)) is None


def test_calibration_flat_csv_with_precomputed_subscores(tmp_path):
    """A flat CSV that already carries sub-score columns + an outcome fits."""
    from neuroviral.calibration.fit import fit
    csv = tmp_path / "posts.csv"
    csv.write_text(
        "video,hook,completion,loopability,peak_end,arousal,retention_pct\n"
        "a.mp4,0.2,0.3,0.2,0.2,0.2,20\n"
        "b.mp4,0.6,0.6,0.5,0.5,0.5,55\n"
        "c.mp4,0.9,0.9,0.8,0.8,0.8,88\n"
    )
    out = tmp_path / "w.json"
    overrides = fit(str(csv), out_path=str(out))
    assert overrides is not None
    assert out.exists()
    assert "platform_weights" in overrides
    assert set(overrides["platform_weights"]["tiktok"]) == {
        "hook", "completion", "loopability", "peak_end", "arousal"}


@has_ffmpeg
def test_calibration_flat_csv_scores_videos(clips, tmp_path):
    """A flat `video,views` CSV with no sub-scores: fit scores each clip."""
    from neuroviral.calibration.fit import fit
    csv = tmp_path / "posts.csv"
    csv.write_text(
        "video,views\n"
        f"{clips['high']},50000\n"
        f"{clips['slow']},1200\n"
        f"{clips['short']},8000\n"
    )
    overrides = fit(str(csv), out_path=str(tmp_path / "w.json"))
    assert overrides is not None
    assert "platform_weights" in overrides
