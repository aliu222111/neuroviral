"""One-pass analysis pipeline: media -> extractors -> brain -> scoring -> report.

This orchestrator ties the spec's stages together. It is deliberately tolerant:
any extractor may degrade to neutral output, and the pipeline records how many
did so (used to widen the confidence band).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .brain import weights as W
from .brain.channels import build_channels, infer_content_type
from .brain.dynamics import derive
from .extractors import audio as audio_x
from .extractors import faces as faces_x
from .extractors import speech as speech_x
from .extractors import text_semantics as text_x
from .extractors import vision as vision_x
from .io import media as media_io
from .report import Result, build_result
from .scoring.platforms import score_all
from .scoring.recommendations import generate
from .utils import TimeGrid

FRAME_FPS = 4.0
MIN_DURATION = 0.5


def analyze(
    video_path: str,
    platforms: Optional[List[str]] = None,
    content_type: str = "auto",
    whisper_model: str = "base",
    weights: Optional[dict] = None,
) -> Result:
    platforms = platforms or ["tiktok", "reels"]
    weights = weights or W.get_weights()

    info = media_io.probe(video_path)
    duration = max(MIN_DURATION, info.duration or MIN_DURATION)
    grid = TimeGrid(duration=duration)

    pipeline_notes: List[str] = list(info.notes)

    # --- decode ---
    wave = media_io.extract_audio(video_path)
    frames = media_io.sample_frames(video_path, fps=FRAME_FPS)
    if wave is None:
        pipeline_notes.append("no audio decoded")
    if not frames:
        pipeline_notes.append("no video frames decoded")

    # --- extractors ---
    a_res = audio_x.extract(wave, grid)
    rms = a_res.series.get("rms_energy")
    s_res = speech_x.extract(wave, grid, rms_series=rms, model_size=whisper_model)
    transcript = getattr(s_res, "_transcript", None)
    t_res = text_x.extract(transcript, grid)
    v_res = vision_x.extract(frames, FRAME_FPS, grid)
    f_res = faces_x.extract(frames, FRAME_FPS, grid)

    extractor_results = {
        "audio": a_res, "speech": s_res, "text": t_res,
        "vision": v_res, "faces": f_res,
    }

    # --- fuse features ---
    features: Dict[str, np.ndarray] = {}
    scalars: Dict = {}
    n_degraded = 0
    for name, r in extractor_results.items():
        features.update(r.series)
        scalars.update(r.scalars)
        if r.degraded:
            n_degraded += 1
        for note in r.notes:
            pipeline_notes.append(f"[{name}] {note}")

    # --- content type ---
    if content_type == "auto":
        content_type = infer_content_type(features, scalars)

    channels = build_channels(features, grid, content_type=content_type, weights=weights)

    first_frame = getattr(v_res, "_first_frame", None)
    last_frame = getattr(v_res, "_last_frame", None)
    have_speech = bool(scalars.get("have_speech", False))
    dynamics = derive(channels, weights=weights,
                      first_frame=first_frame, last_frame=last_frame,
                      features=features, have_speech=have_speech)

    scores = score_all(dynamics, platforms, n_degraded=n_degraded, weights=weights)
    recs = generate(channels, dynamics, scores, have_speech=have_speech)

    return build_result(
        video_path=video_path,
        info=info,
        content_type=content_type,
        features=features,
        scalars=scalars,
        channels=channels,
        dynamics=dynamics,
        scores=scores,
        recommendations=recs,
        n_degraded=n_degraded,
        weights=weights,
        notes=pipeline_notes,
    )
