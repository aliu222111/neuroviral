"""Assemble the structured result and emit JSON / Markdown / terminal output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .brain.channels import CHANNELS, Channels
from .brain.dynamics import Dynamics
from .calibration.schema import CalibrationRecord, SCHEMA_VERSION, features_hash
from .io.media import MediaInfo
from .scoring.platforms import PlatformScore
from .scoring.recommendations import Recommendation

DISCLAIMER = (
    "NeuroViral is a research-informed SIMULATION / PROXY, not a real brain scan. "
    "It maps real audio/visual signals onto a model of neural response grounded in "
    "neuroforecasting research (Falk et al. 2012; Tong et al. 2020, PNAS). Scores "
    "are relative optimization guidance with a confidence band, not guaranteed views."
)

SPARK = " ▁▂▃▄▅▆▇█"


def _sparkline(x: np.ndarray, width: int = 48) -> str:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return ""
    if x.size > width:
        idx = np.linspace(0, x.size - 1, width).astype(int)
        x = x[idx]
    lo, hi = 0.0, 1.0
    norm = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
    return "".join(SPARK[int(v * (len(SPARK) - 1))] for v in norm)


@dataclass
class Result:
    video_path: str
    duration: float
    content_type: str
    channels: Channels
    dynamics: Dynamics
    scores: Dict[str, PlatformScore]
    recommendations: List[Recommendation]
    features: Dict[str, np.ndarray]
    scalars: Dict
    n_degraded: int
    calibrated: bool
    notes: List[str] = field(default_factory=list)
    info: MediaInfo = None

    # ----- calibration record -----
    def calibration_record(self) -> CalibrationRecord:
        ch = self.channels
        rec = CalibrationRecord(
            schema_version=SCHEMA_VERSION,
            video_path=self.video_path,
            duration=round(self.duration, 3),
            content_type=self.content_type,
            features_hash=features_hash(self.features, self.content_type, self.duration),
            channel_means={c: round(float(np.mean(getattr(ch, c))), 4) for c in CHANNELS},
            hook=round(self.dynamics.hook, 4),
            predicted_completion=round(self.dynamics.predicted_completion, 4),
            loopability=round(self.dynamics.loopability, 4),
            peak_end=round(self.dynamics.peak_end, 4),
            arousal=round(self.dynamics.arousal, 4),
            scores={p: {
                "score": s.score, "confidence": s.confidence,
                "low": s.low, "high": s.high, "subscores": s.subscores,
            } for p, s in self.scores.items()},
            degraded_extractors=self.n_degraded,
            calibrated_weights=self.calibrated,
        )
        return rec

    # ----- serialization -----
    def to_dict(self) -> Dict:
        ch = self.channels
        return {
            "video_path": self.video_path,
            "duration": round(self.duration, 3),
            "content_type": self.content_type,
            "disclaimer": DISCLAIMER,
            "scores": {p: {
                "score": s.score, "confidence": s.confidence,
                "band": [s.low, s.high], "subscores": s.subscores,
            } for p, s in self.scores.items()},
            "dynamics": {
                "hook": round(self.dynamics.hook, 4),
                "visual_hook": round(self.dynamics.visual_hook, 4),
                "text_hook": round(self.dynamics.text_hook, 4),
                "audio_hook": round(self.dynamics.audio_hook, 4),
                "predicted_completion": round(self.dynamics.predicted_completion, 4),
                "loopability": round(self.dynamics.loopability, 4),
                "peak_end": round(self.dynamics.peak_end, 4),
                "arousal": round(self.dynamics.arousal, 4),
                "arousal_peak_time": round(self.dynamics.arousal_peak_time, 2),
                "uniqueness": round(self.dynamics.uniqueness, 4),
                "storytelling": round(self.dynamics.storytelling, 4),
                "dropoff_zones": self.dynamics.dropoff_zones,
            },
            "channels": {c: [round(float(v), 4) for v in getattr(ch, c)] for c in CHANNELS},
            "retention_curve": [round(float(v), 4) for v in self.dynamics.retention_curve],
            "times": [round(float(t), 3) for t in ch.times],
            "recommendations": [{
                "title": r.title, "detail": r.detail,
                "timestamp": r.timestamp_label,
                "t_start": r.t_start, "t_end": r.t_end,
                "channel": r.channel, "est_lift": round(r.est_lift, 1),
            } for r in self.recommendations],
            "degraded_extractors": self.n_degraded,
            "calibrated_weights": self.calibrated,
            "notes": self.notes,
            "calibration_record": self.calibration_record().to_dict(),
        }

    # ----- terminal / markdown -----
    def to_terminal(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"  NeuroViral  —  {self.video_path}")
        lines.append(f"  duration {self.duration:.1f}s   content-type: {self.content_type}"
                     + ("   [CALIBRATED]" if self.calibrated else ""))
        lines.append("=" * 60)
        for p, s in self.scores.items():
            bar = "█" * int(s.score / 5)
            lines.append(f"  {p.upper():7s}  {s.score:5.1f}/100  "
                         f"[{s.low:.0f}–{s.high:.0f}]  conf {s.confidence:.2f}  {bar}")
        lines.append("")
        d = self.dynamics
        lines.append("  Sub-scores:")
        lines.append(f"    Hook (0-3s)          {d.hook*100:5.1f}")
        lines.append(f"       ├ visual          {d.visual_hook*100:5.1f}")
        lines.append(f"       ├ text/script     {d.text_hook*100:5.1f}")
        lines.append(f"       └ audio           {d.audio_hook*100:5.1f}")
        lines.append(f"    Predicted completion {d.predicted_completion*100:5.1f}%")
        lines.append(f"    Loopability          {d.loopability*100:5.1f}")
        lines.append(f"    Peak-end / share     {d.peak_end*100:5.1f}")
        lines.append(f"    Arousal              {d.arousal*100:5.1f}")
        lines.append(f"    Uniqueness           {d.uniqueness*100:5.1f}")
        lines.append(f"    Storytelling         {d.storytelling*100:5.1f}")
        lines.append("")
        lines.append("  Neural channels over time (low → high):")
        ch = self.channels
        for c in CHANNELS:
            lines.append(f"    {c:10s} {_sparkline(getattr(ch, c))}")
        lines.append(f"    {'RETENTION':10s} {_sparkline(d.retention_curve)}")
        lines.append("")
        lines.append("  Top recommendations (ranked by est. score lift):")
        for i, r in enumerate(self.recommendations, 1):
            ts = f"[{r.timestamp_label}] " if r.timestamp_label else ""
            lines.append(f"    {i}. (+{r.est_lift:.0f}) {ts}{r.title}")
            lines.append(f"       {r.detail}")
        lines.append("")
        if self.n_degraded:
            lines.append(f"  Note: {self.n_degraded} extractor(s) degraded to neutral "
                         "output (missing model/dep or absent signal).")
        lines.append("  " + DISCLAIMER)
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        d = self.dynamics
        md = [f"# NeuroViral report — `{self.video_path}`", ""]
        md.append(f"*Duration {self.duration:.1f}s · content-type "
                  f"`{self.content_type}`*", )
        md.append("")
        md.append("| Platform | Score | Band | Confidence |")
        md.append("|---|---|---|---|")
        for p, s in self.scores.items():
            md.append(f"| {p.upper()} | {s.score:.1f}/100 | {s.low:.0f}–{s.high:.0f} | {s.confidence:.2f} |")
        md.append("")
        md.append("## Sub-scores")
        md.append(f"- Hook (0-3s): **{d.hook*100:.1f}**")
        md.append(f"- Predicted completion: **{d.predicted_completion*100:.1f}%**")
        md.append(f"- Loopability: **{d.loopability*100:.1f}**")
        md.append(f"- Peak-end / shareability: **{d.peak_end*100:.1f}**")
        md.append(f"- Arousal: **{d.arousal*100:.1f}**")
        md.append("")
        md.append("## Recommendations")
        for i, r in enumerate(self.recommendations, 1):
            ts = f" `[{r.timestamp_label}]`" if r.timestamp_label else ""
            md.append(f"{i}. **{r.title}** (est. +{r.est_lift:.0f}){ts}  \n   {r.detail}")
        md.append("")
        md.append(f"> {DISCLAIMER}")
        return "\n".join(md)


def build_result(
    video_path,
    info,
    content_type,
    features,
    scalars,
    channels,
    dynamics,
    scores,
    recommendations,
    n_degraded,
    weights,
    notes,
) -> Result:
    return Result(
        video_path=str(video_path),
        duration=channels.grid.duration,
        content_type=content_type,
        channels=channels,
        dynamics=dynamics,
        scores=scores,
        recommendations=recommendations,
        features=features,
        scalars=scalars,
        n_degraded=n_degraded,
        calibrated=bool(weights.get("_calibrated", False)),
        notes=notes,
        info=info,
    )
