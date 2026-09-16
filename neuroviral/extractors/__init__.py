"""Feature extractors. Each emits per-timestep series aligned to a TimeGrid.

Every extractor is written to *degrade gracefully*: if its optional heavy
dependency (librosa / faster-whisper / mediapipe / sentence-transformers) is not
installed, or the media lacks the relevant stream, it returns neutral series and
flags ``degraded=True`` with an explanatory note instead of raising.
"""
