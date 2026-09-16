# NeuroViral

A **fully-local** tool that "watches and listens" to a short-form video and predicts how it might perform on TikTok and Instagram Reels. It extracts real audio/visual signals, maps them onto a research-grounded model of neural/attention response, and returns per-platform virality scores plus specific, timestamped fixes.

> ⚠️ **Honest framing — read this first.** NeuroViral is a **research-informed SIMULATION / PROXY, not a real brain scan.** No software can image a hypothetical viewer's brain. What this tool does is extract measurable features from your video and map them onto a model of neural response grounded in published *neuroforecasting* research. Scores are **relative optimization guidance with a confidence band, not guaranteed views.**

Everything runs on your machine. There are **no network/API calls at score time** — private and free per run.

---

## The science it's grounded in

Neuroforecasting studies found that specific brain responses at video onset predict *population-level* view frequency and duration:

- **Falk, Berkman & Lieberman (2012),** *Psychological Science* — a "neural focus group": group-level nucleus accumbens (NAcc, reward/anticipation) ↑ and anterior insula (AIns, aversion) ↓ forecast population media effects.
- **Tong, Chen, Cikara, Zaki, Genevsky, Falk & Knutson (2020),** *PNAS* — brain activity (NAcc ↑, AIns ↓, MPFC) at video onset forecasts aggregate YouTube view frequency/duration above conventional measures.

NeuroViral models those systems as **five neural channels** activated over time by features it can measure locally, then translates their dynamics into platform scores using documented TikTok/Reels ranking behavior (first-3s watch-time is the strongest signal; 80%+ completion = high viral potential; replays/loops, saves, and shares dominate).

| Channel | Brain system | Driven by |
|---|---|---|
| Reward / Anticipation | Nucleus accumbens | curiosity gaps, hook phrasing, novelty, build-ups, payoff timing |
| Attention / Salience | visual/attention networks | cut rate, motion, face presence/size, contrast, on-screen text |
| Emotion / Arousal | amygdala / insula | audio energy, prosody (pitch/pace), expression intensity, sentiment |
| Aversion / Drop-off | anterior insula (inverse) | dead air, slow starts, low motion + monotone, over-long shots |
| Language / Social meaning | MPFC / language areas | semantic novelty, self-reference/relatability, clarity, CTA |

From the channels it derives **hook strength (0–3s)**, a **predicted retention curve / completion %**, **loopability**, and **peak-end shareability**, which become the TikTok and Reels scores.

---

## Install

Requires **Python 3.11+** and **ffmpeg** (a system dependency).

```bash
# ffmpeg (macOS)
brew install ffmpeg

# project
cd neuroviral
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"      # full local-ML stack + dashboard + calibration
```

Install variants (extras are additive):

| Command | Installs | Use when |
|---|---|---|
| `pip install -e ".[all]"` | everything (ml + web + calibrate) | **recommended** — full-fidelity scoring |
| `pip install -e ".[ml]"` | the real extractors (Whisper, librosa, OpenCV, mediapipe, sentence-transformers) | CLI scoring without the dashboard |
| `pip install -e ".[web]"` | FastAPI + uvicorn dashboard | dashboard only |
| `pip install -e .` | **numpy only** | minimal — every extractor runs in degraded/neutral mode, so scores are low-confidence placeholders |

> ⚠️ The bare `pip install -e .` does **not** pull the ML models — with it, all extractors degrade to neutral output and scores carry low confidence. Install `.[ml]` or `.[all]` for real scoring.

**On first real use the ML libraries download several GB of model weights** (Whisper for speech, sentence-embeddings for text novelty, etc.), cached locally thereafter. If a model or optional dependency is still missing, the matching extractor **degrades gracefully** to a neutral signal and the run continues — the report notes how many extractors degraded and lowers its confidence band accordingly.

---

## Usage

### Score a video (CLI)

```bash
neuroviral score clip.mp4 --platform both
```

Prints overall + per-platform scores with a confidence band, sub-scores (hook — broken down into **visual / text / audio** components — predicted completion %, loopability, peak-end, arousal, **uniqueness**, and **storytelling / narrative-arc quality**), a text sparkline of the five channels and the predicted retention curve, and the top ranked, timestamped recommendations.

Useful flags:

| Flag | Purpose |
|---|---|
| `--platform tiktok\|reels\|both` | Which platform(s) to score |
| `--type auto\|talking_head\|edit\|vlog` | Content-type weighting (auto-inferred by default) |
| `--json out.json` | Write the full structured result (also the calibration-ready record) |
| `--markdown out.md` | Write a markdown report |
| `--whisper base\|small\|…` | Whisper model size for transcription |
| `--calibration-log path` | Append this run's calibration record to a log for later fitting |

### Dashboard

```bash
python -m dashboard.server      # serves http://127.0.0.1:8000
```

Upload a clip to see the video player with a synced multi-track brain-activation timeline, platform score gauges, the retention curve with drop-off risk zones highlighted, and a clickable list of fixes that seek the player to each timestamp.

### Other commands

```bash
neuroviral info                 # environment / model availability + config
neuroviral calibrate data.csv   # re-fit weights from your past-post analytics (see below)
```

---

## Design-for-calibration (build now, use later)

Every run emits a **calibration record** (features + channel/dynamics metrics + scores) via `neuroviral/calibration/schema.py`. When you have real analytics from your own posts, feed a table of `video → views / retention / saves / shares` to:

```bash
neuroviral calibrate my_posts.csv
```

This re-fits the coefficients in `neuroviral/brain/weights.py` (ridge/logistic, scikit-learn) so scores learn *your* audience, writing `brain/weights_calibrated.json`, which is layered on top of the research defaults automatically. Until you supply data it is a no-op and research-derived weights are used. All tunable coefficients live in `weights.py`, so calibration only ever touches data — never pipeline code.

---

## Project layout

```
neuroviral/
  io/media.py            ffmpeg decode: audio wav + sampled frames + probe
  extractors/            audio, speech, vision, faces, text_semantics
  brain/                 channels.py (feature→5 channels), dynamics.py, weights.py
  scoring/               platforms.py (scores + confidence), recommendations.py
  calibration/           schema.py, fit.py
  report.py, pipeline.py, cli.py
dashboard/               FastAPI server + single-page timeline UI
tests/                   unit + end-to-end (synthetic clips generated with ffmpeg)
```

## Tests

```bash
pip install -e ".[dev]"   # or: pip install pytest
pytest
```

The suite generates tiny synthetic clips with ffmpeg at test time (no committed binaries) and runs green **without downloading the large ML models** — extractors take their graceful-degradation paths, and unit tests exercise the brain/scoring layers directly. Tests needing ffmpeg skip automatically if it is absent.

---

## Limitations & honesty guarantees

- **Not a real brain scan.** It's a research-informed simulation/proxy; the dashboard and every report say so.
- **No network calls at score time** (privacy + fully-local by design).
- Scores are **relative guidance**, not guaranteed view counts — presented with a confidence band and improvable via calibration.
- Uncalibrated scores reflect general research heuristics, not your specific audience. Calibrate once you have post analytics.

## Sources

- Falk, Berkman & Lieberman (2012), neural focus group — <https://www.academia.edu/2790120/From_neural_responses_to_population_behavior_Neural_focus_group_predicts_population_level_media_effects_Emily_B_Falk_University_of_Michigan>
- Tong et al. (2020), *PNAS*, brain activity forecasts video engagement — <http://web.stanford.edu/~genevsky/files/Tong_PNAS_2020.pdf>
- Wu Tsai Neurosciences Institute, neuroforecasting overview — <https://neuroscience.stanford.edu/news/neuroforecasting-how-brain-activity-can-predict-stock-prices-or-viral-videos>
