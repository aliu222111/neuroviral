"""Extractor unit tests. Cover graceful degradation (no models needed) and,
when ffmpeg is present, real synthetic clips."""
from __future__ import annotations

import numpy as np

from neuroviral.extractors import audio as audio_x
from neuroviral.extractors import faces as faces_x
from neuroviral.extractors import speech as speech_x
from neuroviral.extractors import text_semantics as text_x
from neuroviral.extractors import vision as vision_x
from neuroviral.io.media import AudioWaveform, probe, extract_audio, sample_frames
from neuroviral.utils import TimeGrid
from .conftest import has_ffmpeg


def _grid(dur=4.0):
    return TimeGrid(duration=dur)


# --- graceful degradation (no deps / no signal) ---------------------------

def test_audio_none_degrades():
    grid = _grid()
    res = audio_x.extract(None, grid)
    assert res.degraded
    assert res.series["rms_energy"].shape[0] == grid.n


def test_audio_on_synthetic_waveform():
    grid = _grid(4.0)
    sr = 16000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    # loud first half, silent second half
    sig = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    sig[len(sig) // 2:] = 0.0
    res = audio_x.extract(AudioWaveform(samples=sig, sr=sr), grid)
    rms = res.series["rms_energy"]
    assert rms[: grid.n // 2].mean() > rms[grid.n // 2:].mean()


def test_speech_without_whisper_uses_energy():
    grid = _grid()
    sr = 16000
    sig = np.zeros(sr * 4, dtype=np.float32)  # all silence => dead air
    res = speech_x.extract(AudioWaveform(samples=sig, sr=sr), grid,
                           rms_series=grid.zeros())
    assert res.series["dead_air"].shape[0] == grid.n
    # all-silent => dead_air present somewhere
    assert res.series["dead_air"].max() >= 0.0


def test_vision_no_frames_degrades():
    grid = _grid()
    res = vision_x.extract([], 4.0, grid)
    assert res.degraded
    for k in ("motion", "cut_rate", "color_contrast"):
        assert res.series[k].shape[0] == grid.n


def test_vision_detects_motion_and_cuts():
    grid = _grid(4.0)
    # build 16 frames: static for first half, flto flip (cut) at midpoint
    frames = []
    for i in range(16):
        f = np.zeros((32, 32, 3), dtype=np.uint8)
        if i >= 8:
            f[:] = 255  # hard cut to white + high contrast change
        if i % 2 == 0 and i >= 8:
            f[:16] = 0  # add motion in second half
        frames.append(f)
    res = vision_x.extract(frames, 4.0, grid)
    motion = res.series["motion"]
    assert motion[grid.n // 2:].sum() > 0
    assert res.scalars["n_cuts"] >= 1


def test_faces_no_frames_degrades():
    grid = _grid()
    res = faces_x.extract([], 4.0, grid)
    assert res.degraded
    assert res.series["face_presence"].shape[0] == grid.n


def test_text_semantics_lexicon_no_models():
    from neuroviral.extractors.speech import Transcript, Word
    grid = _grid(6.0)
    words = []
    # "you won't believe" curiosity/hook at t=0; "follow" CTA at end
    for i, tok in enumerate(["you", "won't", "believe", "this", "crazy", "story"]):
        words.append(Word(text=tok, start=i * 0.4, end=i * 0.4 + 0.35))
    words.append(Word(text="follow!", start=5.0, end=5.4))
    tr = Transcript(text="you won't believe this crazy story follow!", words=words)
    res = text_x.extract(tr, grid)
    assert res.series["curiosity"].max() > 0 or res.series["hook_phrase"].max() > 0
    assert res.series["cta"].max() > 0
    assert res.series["sentiment_arousal"].max() > 0  # "crazy" + "!"


def test_text_semantics_empty_degrades():
    grid = _grid()
    from neuroviral.extractors.speech import Transcript
    res = text_x.extract(Transcript(), grid)
    assert res.degraded
    assert res.series["novelty"].shape[0] == grid.n


# --- real ffmpeg-decoded clips --------------------------------------------

@has_ffmpeg
def test_probe_and_decode_real_clip(clips):
    info = probe(clips["high"])
    assert info.duration > 1.0
    assert info.has_video
    frames = sample_frames(clips["high"], fps=4.0)
    assert len(frames) > 2
    wave = extract_audio(clips["high"])
    assert wave is not None and wave.samples.size > 0


@has_ffmpeg
def test_silent_clip_has_no_audio(clips):
    info = probe(clips["short"])
    assert not info.has_audio
    wave = extract_audio(clips["short"])
    assert wave is None  # no audio track
