"""Calibration record schema.

Every NeuroViral run emits one :class:`CalibrationRecord`: a stable, hashable
snapshot of the derived channel/dynamics metrics and the scores produced. Later,
you append the *real* outcome (views / retention / saves / shares) to these
records and feed them to ``calibration/fit.py`` to re-fit ``brain/weights.py``.

The record is intentionally flat and JSON-serializable so it can live in a CSV
or a JSONL log next to your post analytics.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

SCHEMA_VERSION = 1


@dataclass
class Outcome:
    """Real post analytics, filled in AFTER posting (all optional)."""
    views: Optional[int] = None
    retention_pct: Optional[float] = None   # 0-100 real average watch-through
    completion_rate: Optional[float] = None
    saves: Optional[int] = None
    shares: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    platform: Optional[str] = None
    posted_at: Optional[str] = None


@dataclass
class CalibrationRecord:
    schema_version: int
    video_path: str
    duration: float
    content_type: str
    features_hash: str
    # derived channel summaries (mean activation per channel)
    channel_means: Dict[str, float] = field(default_factory=dict)
    # derived dynamics
    hook: float = 0.0
    predicted_completion: float = 0.0
    loopability: float = 0.0
    peak_end: float = 0.0
    arousal: float = 0.0
    # produced scores per platform {platform: {score, confidence, subscores...}}
    scores: Dict[str, Dict] = field(default_factory=dict)
    degraded_extractors: int = 0
    calibrated_weights: bool = False
    # populated later
    outcome: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def features_hash(features: Dict, content_type: str, duration: float) -> str:
    """Stable hash of the feature-series shape + content-type + duration.

    Uses coarse rounded means so tiny numerical noise doesn't change the hash,
    but different videos produce different hashes.
    """
    h = hashlib.sha256()
    h.update(f"{SCHEMA_VERSION}|{content_type}|{round(duration, 2)}".encode())
    for name in sorted(features.keys()):
        arr = features[name]
        try:
            import numpy as np
            m = float(np.mean(arr)) if len(arr) else 0.0
            s = float(np.std(arr)) if len(arr) else 0.0
        except Exception:
            m, s = 0.0, 0.0
        h.update(f"{name}:{round(m, 4)}:{round(s, 4)}".encode())
    return h.hexdigest()[:16]
