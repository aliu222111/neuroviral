"""Re-fit weights from labeled past-post data. No-op until data exists.

Given a JSONL/CSV of :class:`CalibrationRecord`-shaped rows with a filled
``outcome`` (real views/retention), this fits a small regularized model mapping
the derived sub-scores to the observed outcome, and writes multiplicative
adjustments into ``brain/weights_calibrated.json`` (consumed by
``weights.get_weights()``). Everything upstream is untouched — calibration only
ever changes data.

Requires scikit-learn (``pip install -e ".[calibrate]"``). With zero labeled
rows this function returns ``None`` and changes nothing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

from ..brain import weights as W

SUBSCORE_KEYS = ["hook", "completion", "loopability", "peak_end", "arousal"]


def _load_rows(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text().strip()
    if not text:
        return []
    rows: List[dict] = []
    if p.suffix.lower() == ".csv":
        for r in csv.DictReader(text.splitlines()):
            rows.append(dict(r))
    else:  # jsonl or json array
        if text.lstrip().startswith("["):
            rows = json.loads(text)
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _as_dict(v):
    """Coerce a possibly-JSON-string cell (from CSV) into a dict, else {}."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


def _num(v) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row_outcome(row: dict) -> Optional[float]:
    """Target value from a nested ``outcome`` dict or flat columns.

    Preference: retention_pct > completion_rate > views.
    """
    outcome = _as_dict(row.get("outcome"))
    for key in ("retention_pct", "completion_rate", "views"):
        val = _num(outcome.get(key) if key in outcome else row.get(key))
        if val is not None:
            return val
    return None


def _row_platform(row: dict) -> Optional[str]:
    outcome = _as_dict(row.get("outcome"))
    return outcome.get("platform") or row.get("platform") or None


def _row_subscores(row: dict, platform: Optional[str]) -> Optional[List[float]]:
    """Sub-scores already present in the row (nested ``scores`` or flat columns).

    Returns ``None`` when the row carries no sub-scores (caller then scores the
    video to obtain them).
    """
    scores = _as_dict(row.get("scores"))
    subs = None
    if platform and platform in scores:
        subs = _as_dict(scores[platform]).get("subscores")
    if not subs:
        for pdata in scores.values():
            cand = _as_dict(pdata).get("subscores")
            if cand:
                subs = cand
                break
    if subs:
        return [float(_num(subs.get(k)) or 0.0) for k in SUBSCORE_KEYS]

    # Flat columns: hook / completion|predicted_completion / loopability / peak_end / arousal.
    flat = {}
    for k in SUBSCORE_KEYS:
        src = row.get("predicted_completion") if k == "completion" and "completion" not in row else row.get(k)
        flat[k] = _num(src)
    if any(v is not None for v in flat.values()):
        return [flat[k] or 0.0 for k in SUBSCORE_KEYS]
    return None


def _resolve_video(row: dict, base_dir: Optional[Path]) -> Optional[str]:
    raw = row.get("video") or row.get("video_path")
    if not raw:
        return None
    for cand in ([Path(raw)] + ([base_dir / raw] if base_dir else [])):
        if cand.exists():
            return str(cand)
    return None


_SCORE_CACHE: dict = {}


def _score_video(path: str) -> Optional[List[float]]:
    """Run the pipeline on a past video to recover its sub-scores for fitting."""
    if path in _SCORE_CACHE:
        return _SCORE_CACHE[path]
    try:
        from ..pipeline import analyze
        d = analyze(path).dynamics
        subs = [d.hook, d.predicted_completion, d.loopability, d.peak_end, d.arousal]
    except Exception:
        subs = None
    _SCORE_CACHE[path] = subs
    return subs


def _extract_xy(rows: List[dict], platform: Optional[str], base_dir: Optional[Path] = None):
    X, y = [], []
    for row in rows:
        target = _row_outcome(row)
        if target is None:
            continue
        row_plat = _row_platform(row)
        if platform and row_plat not in (None, platform):
            continue
        subs = _row_subscores(row, platform)
        if subs is None:
            video = _resolve_video(row, base_dir)
            if video is not None:
                subs = _score_video(video)
        if subs is None:
            continue
        X.append(subs)
        y.append(float(target))
    return X, y


def fit(
    data_path: str,
    platform: Optional[str] = None,
    out_path: Optional[str] = None,
    alpha: float = 1.0,
) -> Optional[dict]:
    """Fit sub-score -> outcome and persist platform-weight overrides.

    Returns the override dict written (or ``None`` if there was no usable data).
    """
    rows = _load_rows(data_path)
    base_dir = Path(data_path).resolve().parent
    X, y = _extract_xy(rows, platform, base_dir=base_dir)
    if len(X) < 3:
        # Not enough labeled data — remain a research-derived no-op.
        return None

    try:
        import numpy as np
        from sklearn.linear_model import Ridge
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn required for calibration: pip install -e '.[calibrate]'"
        ) from exc

    Xn = np.asarray(X, dtype=float)
    yn = np.asarray(y, dtype=float)
    model = Ridge(alpha=alpha, positive=True)
    model.fit(Xn, yn)
    coefs = np.clip(model.coef_, 1e-6, None)
    if coefs.sum() <= 0:
        return None
    norm = coefs / coefs.sum()

    platforms = [platform] if platform else ["tiktok", "reels"]
    overrides = {"platform_weights": {}}
    base = W.PLATFORM_WEIGHTS
    for plat in platforms:
        learned = {k: float(round(w, 4)) for k, w in zip(SUBSCORE_KEYS, norm)}
        # blend learned weights with research priors (avoid overfitting tiny data)
        blend = 0.5
        merged = {}
        for k in SUBSCORE_KEYS:
            merged[k] = round(blend * learned[k] + (1 - blend) * base[plat][k], 4)
        overrides["platform_weights"][plat] = merged

    target = out_path or str(Path(W.__file__).with_name("weights_calibrated.json"))
    Path(target).write_text(json.dumps(overrides, indent=2))
    return overrides
