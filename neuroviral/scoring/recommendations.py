"""Rule engine -> timestamped, specific fix suggestions, ranked by score lift.

Each rule inspects the channels/dynamics, and when it fires emits a
``Recommendation`` with a concrete timestamp/range, an explanation grounded in
the neural-channel model, and an estimated score lift used for ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..brain.channels import Channels
from ..brain.dynamics import Dynamics
from ..utils import window_mean


@dataclass
class Recommendation:
    title: str
    detail: str
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    channel: str = ""
    est_lift: float = 0.0  # estimated 0-100 score points

    @property
    def timestamp_label(self) -> str:
        if self.t_start is None:
            return ""
        if self.t_end is None or abs(self.t_end - self.t_start) < 0.25:
            return f"{self.t_start:.1f}s"
        return f"{self.t_start:.1f}-{self.t_end:.1f}s"


def _longest_zone(mask: np.ndarray, times: np.ndarray):
    best = (0, -1, -1)
    i = 0
    n = mask.size
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) > best[0]:
                best = (j - i, i, j - 1)
            i = j
        else:
            i += 1
    if best[2] < 0:
        return None
    return times[best[1]], times[best[2]]


def generate(
    ch: Channels,
    dyn: Dynamics,
    scores: Dict,
    max_recs: int = 5,
    have_speech: bool = True,
) -> List[Recommendation]:
    grid = ch.grid
    times = grid.times
    recs: List[Recommendation] = []

    hook_win = min(grid.n, grid.index_of(3.0) + 1)

    # 1. Weak hook.
    if dyn.hook < 0.5:
        recs.append(Recommendation(
            title="Strengthen the first 3 seconds",
            detail=(f"Hook activation is low ({dyn.hook:.2f}). The first 3s drive "
                    "the watch-time signal platforms weight most. Open with motion, "
                    "a bold on-screen claim, or a curiosity gap ('wait for it…')."),
            t_start=0.0, t_end=3.0, channel="reward+attention",
            est_lift=(0.6 - dyn.hook) * 30,
        ))

    # 1b. Weak VISUAL hook.
    if dyn.visual_hook < 0.45:
        recs.append(Recommendation(
            title="Open with a stronger visual",
            detail=(f"Visual hook is weak ({dyn.visual_hook:.2f}) in the first 3s. "
                    "Lead with motion, a fast first cut, a big face, or a bold "
                    "on-screen text card — the eye commits before the ear does."),
            t_start=0.0, t_end=3.0, channel="attention",
            est_lift=(0.55 - dyn.visual_hook) * 26,
        ))

    # 1c. Weak scripted/TEXT hook (only when there is speech to fix).
    if have_speech and dyn.text_hook < 0.4:
        recs.append(Recommendation(
            title="Front-load a curiosity-gap line",
            detail=(f"Scripted hook is weak ({dyn.text_hook:.2f}). Say the payoff-"
                    "promise in the first sentence — a curiosity gap ('wait for "
                    "it…', 'here's why…') or a 'you/we' line lands the verbal hook "
                    "before viewers scroll."),
            t_start=0.0, t_end=3.0, channel="reward+language",
            est_lift=(0.5 - dyn.text_hook) * 22,
        ))

    # 1d. Weak AUDIO hook.
    if dyn.audio_hook < 0.4:
        recs.append(Recommendation(
            title="Fix the opening audio energy",
            detail=(f"Audio hook is weak ({dyn.audio_hook:.2f}) — likely dead air, "
                    "a slow start, or monotone delivery in the first 3s. Start on a "
                    "loud beat/first word, cut pre-roll silence, and vary pitch."),
            t_start=0.0, t_end=3.0, channel="emotion+aversion",
            est_lift=(0.5 - dyn.audio_hook) * 20,
        ))

    # 2. Slow / dead-air start.
    early_av = float(np.mean(ch.aversion[:hook_win])) if hook_win else 0.0
    if early_av > 0.4:
        recs.append(Recommendation(
            title="Cut the slow intro",
            detail=(f"Aversion (drop-off risk) is high in the opening "
                    f"({early_av:.2f}). Trim dead air/setup so the payoff or "
                    "strongest visual lands in the first second."),
            t_start=0.0, t_end=min(3.0, grid.duration), channel="aversion",
            est_lift=early_av * 22,
        ))

    # 3. Dead-air zones anywhere.
    dead = ch.aversion >= 0.55
    zone = _longest_zone(dead, times)
    # A genuine ~2s dead-air block only survives channel smoothing as a ~0.5s
    # span above the 0.55 threshold, so gate on 0.4s (the threshold itself keeps
    # this from firing on ordinary content).
    if zone is not None and (zone[1] - zone[0]) >= 0.4:
        # A sustained drop-off zone is a concrete, timestamped, high-severity fix
        # (the top scroll trigger), so rank it by how deep AND long it is.
        zmask = (times >= zone[0]) & (times <= zone[1])
        depth = float(np.mean(ch.aversion[zmask])) if zmask.any() else 0.6
        recs.append(Recommendation(
            title="Tighten a drop-off zone",
            detail=(f"Sustained aversion between {zone[0]:.1f}s and {zone[1]:.1f}s "
                    "(dead air / low motion / monotone). Tighten the cut, add "
                    "b-roll, or inject a pattern interrupt here."),
            t_start=round(zone[0], 1), t_end=round(zone[1], 1), channel="aversion",
            est_lift=min(22.0, depth * 20.0 + (zone[1] - zone[0]) * 4.0),
        ))

    # 4. Flat arousal.
    if dyn.arousal < 0.4:
        recs.append(Recommendation(
            title="Raise emotional arousal",
            detail=(f"Peak arousal is muted ({dyn.arousal:.2f}). Vary vocal "
                    "prosody/pace, add energetic music, or a reaction beat — flat "
                    "arousal predicts scroll-away and kills shares."),
            channel="emotion", est_lift=(0.5 - dyn.arousal) * 20,
        ))

    # 5. Low loopability.
    if dyn.loopability < 0.45:
        recs.append(Recommendation(
            title="Make it loop",
            detail=(f"Loopability is low ({dyn.loopability:.2f}). Match the last "
                    "~1s to the opening frame/audio, or end on the unresolved "
                    "setup so replays feel seamless."),
            t_start=round(max(0.0, grid.duration - 1.0), 1),
            t_end=round(grid.duration, 1), channel="reward", est_lift=12,
        ))

    # 6. Weak ending (peak-end).
    end_arousal = float(np.mean(ch.emotion[-max(1, grid.index_of(2.0)):])) if grid.n else 0.0
    if end_arousal < 0.35 and dyn.peak_end < 0.5:
        recs.append(Recommendation(
            title="Land a stronger ending",
            detail=(f"The ending is low-energy (peak-end {dyn.peak_end:.2f}). The "
                    "final beat disproportionately drives saves/shares — end on the "
                    "emotional peak, a punchline, or a clear CTA."),
            t_start=round(max(0.0, grid.duration - 2.0), 1),
            t_end=round(grid.duration, 1), channel="emotion+language", est_lift=10,
        ))

    # 7. Low language/clarity or missing CTA.
    lang_mean = float(np.mean(ch.language)) if grid.n else 0.0
    if lang_mean < 0.35:
        recs.append(Recommendation(
            title="Sharpen the message / add relatability",
            detail=(f"Social-meaning activation is low ({lang_mean:.2f}). Add a "
                    "self-referential 'you/we' framing, a clearer takeaway, or a "
                    "CTA — MPFC self-relevance drives sharing intention."),
            channel="language", est_lift=8,
        ))

    # 8. Low uniqueness — generic content.
    if dyn.uniqueness < 0.4:
        recs.append(Recommendation(
            title="Say something the feed hasn't seen",
            detail=(f"Uniqueness is low ({dyn.uniqueness:.2f}). The script/visuals "
                    "read as familiar, so the novelty-reward signal never spikes. "
                    "Lead with a counter-intuitive claim, a specific number, or a "
                    "personal detail no one else can copy."),
            channel="reward", est_lift=(0.5 - dyn.uniqueness) * 18,
        ))

    # 9. Visually repetitive (uniqueness dragged down by flat visuals).
    if dyn.uniqueness < 0.5 and float(np.std(ch.attention)) < 0.08:
        recs.append(Recommendation(
            title="Break the visual sameness",
            detail=("Every shot looks alike, so the video feels repetitive even if "
                    "the words are fresh. Vary framing, add a location/angle change "
                    "or a b-roll insert every few seconds to keep salience alive."),
            channel="attention", est_lift=8,
        ))

    # 10. Weak narrative arc.
    if dyn.storytelling < 0.4:
        recs.append(Recommendation(
            title="Give it a beginning, build, and payoff",
            detail=(f"Storytelling is flat ({dyn.storytelling:.2f}). Reward barely "
                    "builds and the ending doesn't resolve, so there's no arc pulling "
                    "viewers through. Set up a question early, raise the stakes in the "
                    "middle, and deliver a clear payoff in the last few seconds."),
            channel="reward+emotion", est_lift=(0.5 - dyn.storytelling) * 22,
        ))

    # 11. Missing payoff at the end.
    end_reward = window_mean(ch.reward, grid, grid.duration - 2.0, grid.duration)
    if dyn.storytelling < 0.5 and end_reward < 0.35:
        recs.append(Recommendation(
            title="Deliver the payoff you promised",
            detail=("The arc builds but never resolves — reward is low at the finish. "
                    "End on the answer, reveal, or 'that's why…' beat so the story "
                    "closes. Unresolved arcs read as clickbait and kill re-shares."),
            t_start=round(max(0.0, grid.duration - 2.0), 1),
            t_end=round(grid.duration, 1), channel="reward", est_lift=12,
        ))

    # 12. Emotionally monotone (arc collapses).
    emo_range = float(np.percentile(ch.emotion, 90) - np.percentile(ch.emotion, 10)) if grid.n else 0.0
    if dyn.storytelling < 0.5 and emo_range < 0.12:
        recs.append(Recommendation(
            title="Add emotional highs and lows",
            detail=(f"Emotion stays flat (range {emo_range:.2f}) start to finish, so "
                    "even a well-structured story feels lifeless. Contrast a calm "
                    "setup against an energetic reveal — variation is what makes an "
                    "arc feel like a story."),
            channel="emotion", est_lift=10,
        ))

    recs.sort(key=lambda r: r.est_lift, reverse=True)
    if not recs:
        recs.append(Recommendation(
            title="Solid across the board",
            detail=("No major weaknesses detected by the model. Marginal gains: "
                    "test a punchier first frame and a tighter loop."),
            est_lift=2,
        ))
    return recs[:max_recs]
