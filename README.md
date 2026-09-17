# NeuroViral

A fully local tool that watches and listens to a short-form video and predicts how it might perform on TikTok and Instagram Reels. It extracts audio and visual signals, maps them onto a model of neural attention response taken from published research, and returns per-platform virality scores along with specific, timestamped fixes.

> **Read this before anything else.** NeuroViral is a simulation, not a brain scan. No software can image a hypothetical viewer's brain. What it actually does is pull measurable features out of your video and map them onto a model of neural response drawn from published neuroforecasting work. The scores are relative optimization guidance with a confidence band attached. They are not a prediction of how many views you will get.

Everything runs on your machine. Nothing touches the network at score time, so each run is private and costs nothing.

---

## The science it's built on

Neuroforecasting studies found that specific brain responses at video onset predict view frequency and duration at the population level:

- Falk, Berkman & Lieberman (2012), *Psychological Science*. A "neural focus group": group-level nucleus accumbens (NAcc, reward/anticipation) up and anterior insula (AIns, aversion) down forecast population media effects.
- Tong, Chen, Cikara, Zaki, Genevsky, Falk & Knutson (2020), *PNAS*. Brain activity at video onset (NAcc up, AIns down, MPFC) forecasts aggregate YouTube view frequency and duration better than conventional measures do.

NeuroViral models those systems as five neural channels, activated over time by features it can measure locally, then translates their dynamics into platform scores using documented TikTok and Reels ranking behavior: first-3s watch time is the strongest signal, 80%+ completion indicates high viral potential, and replays, saves and shares dominate the rest.

| Channel | Brain system | Driven by |
|---|---|---|
| Reward / Anticipation | Nucleus accumbens | curiosity gaps, hook phrasing, novelty, build-ups, payoff timing |
| Attention / Salience | visual/attention networks | cut rate, motion, face presence/size, contrast, on-screen text |
| Emotion / Arousal | amygdala / insula | audio energy, prosody (pitch/pace), expression intensity, sentiment |
| Aversion / Drop-off | anterior insula (inverse) | dead air, slow starts, low motion + monotone, over-long shots |
| Language / Social meaning | MPFC / language areas | semantic novelty, self-reference/relatability, clarity, CTA |

From those channels it derives hook strength over the first 3 seconds, a predicted retention curve and completion percentage, loopability, and peak-end shareability. Those become the TikTok and Reels scores.

---

## Install

Requires Python 3.11+ and ffmpeg, which is a system dependency.

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
| `pip install -e ".[all]"` | everything (ml + web + calibrate) | recommended, for full-fidelity scoring |
| `pip install -e ".[ml]"` | the real extractors (Whisper, librosa, OpenCV, mediapipe, sentence-transformers) | CLI scoring without the dashboard |
| `pip install -e ".[web]"` | FastAPI + uvicorn dashboard | dashboard only |
| `pip install -e .` | numpy only | minimal, and every extractor runs in degraded mode, so scores are low-confidence placeholders |

> The bare `pip install -e .` does not pull the ML models. Without them every extractor falls back to neutral output and the scores carry low confidence, so install `.[ml]` or `.[all]` if you want real scoring.

The first real run downloads several GB of model weights (Whisper for speech, sentence-embeddings for text novelty, and so on), which are cached locally afterwards. If a model or an optional dependency is still missing, the matching extractor degrades to a neutral signal and the run carries on. The report tells you how many extractors degraded and widens its confidence band to match.

---

## Usage

### Score a video (CLI)

```bash
neuroviral score clip.mp4 --platform both
```

This prints the overall and per-platform scores with a confidence band, then the sub-scores: hook strength (itself broken into visual, text and audio components), predicted completion percentage, loopability, peak-end, arousal, uniqueness, and narrative-arc quality. Below that comes a text sparkline of the five channels alongside the predicted retention curve, and finally the top-ranked recommendations with timestamps.

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

Upload a clip and you get the video player with a synced multi-track brain-activation timeline, platform score gauges, the retention curve with drop-off risk zones marked, and a clickable list of fixes that seeks the player to each timestamp.

### Other commands

```bash
neuroviral info                 # environment / model availability + config
neuroviral calibrate data.csv   # re-fit weights from your past-post analytics (see below)
```

---

## Design for calibration (build now, use later)

Every run emits a calibration record (features, channel and dynamics metrics, scores) through `neuroviral/calibration/schema.py`. Once you have real analytics from your own posts, feed a table of `video → views / retention / saves / shares` to:

```bash
neuroviral calibrate my_posts.csv
```

That re-fits the coefficients in `neuroviral/brain/weights.py` using ridge or logistic regression from scikit-learn, so the scores learn your audience rather than a generic one. Results are written to `brain/weights_calibrated.json` and layered on top of the research defaults automatically. Until you supply data it does nothing and the research-derived weights stand. Every tunable coefficient lives in `weights.py`, which keeps calibration confined to data rather than pipeline code.

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

The suite generates tiny synthetic clips with ffmpeg at test time, so no binaries are committed, and it runs green without downloading the large ML models: extractors take their degradation paths and the unit tests exercise the brain and scoring layers directly. Tests that need ffmpeg skip themselves if it isn't installed.

---

## Limitations

It is not a brain scan. It is a simulation built on research, and the dashboard and every report say so.

Nothing goes over the network at score time, by design.

The scores are relative guidance rather than predicted view counts, they come with a confidence band, and calibration is what improves them.

Until you calibrate, the scores reflect general research heuristics rather than your specific audience, so treat them accordingly.

## Sources

- Falk, Berkman & Lieberman (2012), neural focus group: <https://www.academia.edu/2790120/From_neural_responses_to_population_behavior_Neural_focus_group_predicts_population_level_media_effects_Emily_B_Falk_University_of_Michigan>
- Tong et al. (2020), *PNAS*, brain activity forecasts video engagement: <http://web.stanford.edu/~genevsky/files/Tong_PNAS_2020.pdf>
- Wu Tsai Neurosciences Institute, neuroforecasting overview: <https://neuroscience.stanford.edu/news/neuroforecasting-how-brain-activity-can-predict-stock-prices-or-viral-videos>
