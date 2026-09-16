"""Speech features: transcript + word timestamps, speech rate, dead-air.

Uses faster-whisper when installed (lazy, model cached after first download).
Without it, dead-air and slow-start are still derived from the audio energy
envelope, and the transcript is empty (text-semantics then degrades too).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ..io.media import AudioWaveform
from ..utils import ExtractResult, TimeGrid, events_to_rate, safe_norm, smooth

_SILENCE_RMS = 0.06  # normalized-energy threshold treated as "quiet"
_MODEL_CACHE = {}


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    text: str = ""
    words: List[Word] = field(default_factory=list)


def _load_whisper(model_size: str = "base"):
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]
    from faster_whisper import WhisperModel  # type: ignore

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _MODEL_CACHE[model_size] = model
    return model


def transcribe(wave: AudioWaveform, model_size: str = "base") -> Transcript:
    """Run faster-whisper. Raises if the dep is missing (caller handles)."""
    import tempfile
    import wave as wavemod

    model = _load_whisper(model_size)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        pcm = np.clip(wave.samples, -1, 1)
        pcm16 = (pcm * 32767).astype(np.int16)
        with wavemod.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(wave.sr)
            wf.writeframes(pcm16.tobytes())
        segments, _ = model.transcribe(path, word_timestamps=True)
        words: List[Word] = []
        texts: List[str] = []
        for seg in segments:
            texts.append(seg.text)
            for w in (seg.words or []):
                words.append(Word(text=w.word, start=float(w.start), end=float(w.end)))
        return Transcript(text=" ".join(t.strip() for t in texts).strip(), words=words)
    finally:
        import os
        try:
            os.unlink(path)
        except OSError:
            pass


def extract(
    wave: Optional[AudioWaveform],
    grid: TimeGrid,
    rms_series: Optional[np.ndarray] = None,
    model_size: str = "base",
) -> ExtractResult:
    res = ExtractResult()
    if rms_series is None:
        rms_series = grid.zeros()
    rms_series = np.asarray(rms_series, dtype=float)

    transcript = Transcript()
    have_speech = False
    if wave is not None and wave.samples.size > 0:
        try:
            transcript = transcribe(wave, model_size=model_size)
            have_speech = len(transcript.words) > 0
            if not have_speech:
                res.note("no speech detected in audio", degraded=False)
        except Exception as exc:  # dep missing or model failed
            res.note(f"faster-whisper unavailable ({type(exc).__name__}); "
                     "using energy-only speech features", degraded=True)
    else:
        res.note("no audio for speech", degraded=True)

    # Speech rate (words/sec) per bin.
    if have_speech:
        centers = [(w.start + w.end) / 2 for w in transcript.words]
        rate = events_to_rate(centers, grid)
        res.series["speech_rate"] = safe_norm(smooth(rate, 3))
        # word presence mask
        word_mask = np.zeros(grid.n, dtype=bool)
        for w in transcript.words:
            i0 = grid.index_of(w.start)
            i1 = grid.index_of(w.end)
            word_mask[i0:i1 + 1] = True
    else:
        res.series["speech_rate"] = grid.zeros()
        word_mask = np.zeros(grid.n, dtype=bool)

    # Dead air: quiet AND (no speech words there). Where no transcript, rely on
    # energy alone.
    quiet = rms_series < _SILENCE_RMS
    dead = quiet & (~word_mask if have_speech else np.ones(grid.n, dtype=bool))
    dead_series = smooth(dead.astype(float), 3)
    res.series["dead_air"] = safe_norm(dead_series) if dead_series.max() > 0 else dead_series

    # Slow start: high if the first ~3s is quiet / speechless / low energy.
    slow = grid.zeros()
    n_early = max(1, grid.index_of(3.0) + 1)
    early_quiet = float(np.mean(quiet[:n_early])) if n_early else 0.0
    early_energy = float(np.mean(rms_series[:n_early])) if n_early else 0.0
    slow_val = np.clip(0.7 * early_quiet + 0.3 * (1.0 - early_energy), 0, 1)
    # ramp that decays after the hook window
    ramp = np.clip(1.0 - (grid.times / max(1e-6, 3.0)), 0, 1)
    res.series["slow_start"] = slow_val * ramp

    # Clarity: fraction of duration with intelligible speech coverage.
    if have_speech:
        coverage = float(np.mean(word_mask))
        clarity_val = np.clip(0.35 + 0.65 * coverage, 0, 1)
    else:
        clarity_val = 0.5  # neutral: could be a no-speech edit
    res.series["clarity"] = grid.full(clarity_val)

    res.scalars["transcript"] = transcript.text
    res.scalars["word_count"] = len(transcript.words)
    res.scalars["have_speech"] = have_speech
    res._transcript = transcript  # type: ignore[attr-defined]
    return res
