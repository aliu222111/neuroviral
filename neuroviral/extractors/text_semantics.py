"""Text-semantic features from the transcript.

Curiosity gaps, hook phrasing, self-reference/relatability, CTAs and arousal
words come from curated lexicons/regex (always available, pure-python). Semantic
novelty uses sentence-transformers when installed and falls back to a lexical
token-novelty measure otherwise.
"""
from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from ..utils import ExtractResult, TimeGrid, events_to_rate, safe_norm, smooth

# --- lexicons --------------------------------------------------------------
CURIOSITY_PHRASES = [
    "wait for it", "you won't believe", "you wont believe", "watch this",
    "here's why", "heres why", "the secret", "what happens next", "keep watching",
    "but then", "little did", "turns out", "the reason", "guess what",
    "i didn't expect", "i didnt expect", "you need to see", "watch till the end",
    "stay till the end", "the crazy part", "here's the thing", "heres the thing",
]
HOOK_PHRASES = [
    "you won't believe", "you wont believe", "stop scrolling", "this is why",
    "nobody talks about", "here's how", "heres how", "did you know",
    "the truth about", "i can't believe", "i cant believe", "watch this",
    "wait for it", "pov", "story time", "storytime",
]
SELF_REF = {
    "you", "your", "you're", "youre", "we", "us", "our", "everyone", "nobody",
    "anyone", "relatable", "yourself", "y'all", "yall",
}
CTA_PHRASES = [
    "follow", "like", "comment", "share", "subscribe", "link in bio",
    "save this", "let me know", "tag someone", "part 2", "part two",
    "follow for more", "hit follow", "drop a comment", "check the link",
]
AROUSAL_WORDS = {
    "crazy", "insane", "shocking", "amazing", "wow", "unbelievable", "omg",
    "never", "best", "worst", "hate", "love", "wild", "epic", "incredible",
    "ridiculous", "mind-blowing", "mindblowing", "unreal", "shook", "obsessed",
    "literally", "actually", "terrifying", "hilarious", "genius", "perfect",
}
PAYOFF_WORDS = {
    "because", "so", "therefore", "result", "answer", "reveal", "revealed",
    "finally", "boom", "voila", "ta-da", "that's why", "thats why", "the trick",
}


def _norm_token(t: str) -> str:
    return re.sub(r"[^a-z0-9']", "", t.lower())


def _phrase_hits(tokens, times, phrases) -> List[float]:
    hits = []
    norm = [_norm_token(t) for t in tokens]
    for phrase in phrases:
        parts = phrase.split()
        k = len(parts)
        pnorm = [_norm_token(p) for p in parts]
        for i in range(0, max(0, len(norm) - k + 1)):
            if norm[i:i + k] == pnorm:
                hits.append(float(np.mean(times[i:i + k])))
    return hits


def _word_hits(tokens, times, vocab) -> List[float]:
    hits = []
    for tok, t in zip(tokens, times):
        if _norm_token(tok) in vocab:
            hits.append(float(t))
    return hits


def extract(transcript, grid: TimeGrid) -> ExtractResult:
    res = ExtractResult()
    words = getattr(transcript, "words", None) or []
    text = getattr(transcript, "text", "") or ""

    if not words:
        for name in ("curiosity", "hook_phrase", "self_reference", "cta",
                     "sentiment_arousal", "payoff"):
            res.series[name] = grid.zeros()
        res.series["novelty"] = grid.full(0.3)
        res.note("no transcript, text-semantic features neutral", degraded=True)
        return res

    tokens = [w.text for w in words]
    times = np.asarray([(w.start + w.end) / 2 for w in words])

    res.series["curiosity"] = safe_norm(smooth(events_to_rate(
        _phrase_hits(tokens, times, CURIOSITY_PHRASES), grid), 3))
    res.series["hook_phrase"] = _front_loaded(
        _phrase_hits(tokens, times, HOOK_PHRASES), grid)
    res.series["self_reference"] = safe_norm(smooth(events_to_rate(
        _word_hits(tokens, times, SELF_REF), grid), 3))
    res.series["cta"] = safe_norm(smooth(events_to_rate(
        _phrase_hits(tokens, times, CTA_PHRASES) +
        _word_hits(tokens, times, {"follow", "subscribe", "comment", "share", "like"}),
        grid), 3))

    # Arousal: arousal-word density + exclamation marks.
    arousal_times = _word_hits(tokens, times, AROUSAL_WORDS)
    excl = sum(t.count("!") for t in tokens)
    arousal_series = events_to_rate(arousal_times, grid)
    if excl:
        for w in words:
            if "!" in w.text:
                arousal_series[grid.index_of((w.start + w.end) / 2)] += 1.0
    res.series["sentiment_arousal"] = safe_norm(smooth(arousal_series, 3))

    # Payoff: payoff words, weighted toward the back half.
    payoff_times = _word_hits(tokens, times, PAYOFF_WORDS)
    payoff = events_to_rate(payoff_times, grid)
    backweight = np.clip(grid.times / max(1e-6, grid.duration), 0.2, 1.0)
    res.series["payoff"] = safe_norm(smooth(payoff * backweight, 3))

    # Novelty.
    res.series["novelty"] = _novelty(words, times, grid, res)

    res.scalars["n_words"] = len(words)
    res.scalars["curiosity_hits"] = int(np.sum(res.series["curiosity"] > 0))
    return res


def _front_loaded(hit_times: List[float], grid: TimeGrid) -> np.ndarray:
    """Hook phrases matter most early; emphasize hits in the first 3s."""
    series = grid.zeros()
    for t in hit_times:
        i = grid.index_of(t)
        weight = 1.0 if t <= 3.0 else 0.5
        series[i] += weight
    return safe_norm(smooth(series, 3))


def _novelty(words, times, grid: TimeGrid, res: ExtractResult) -> np.ndarray:
    # Split transcript into ~2s chunks; novelty = 1 - similarity to prior chunks.
    chunk_s = 2.0
    chunks = []
    chunk_times = []
    cur, cur_start = [], words[0].start
    for w in words:
        cur.append(w.text)
        if w.end - cur_start >= chunk_s:
            chunks.append(" ".join(cur))
            chunk_times.append((cur_start + w.end) / 2)
            cur, cur_start = [], w.end
    if cur:
        chunks.append(" ".join(cur))
        chunk_times.append((cur_start + words[-1].end) / 2)
    if not chunks:
        return grid.full(0.3)

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = _get_st_model()
        embs = model.encode(chunks, normalize_embeddings=True)
        novelty_vals = [1.0]
        for i in range(1, len(embs)):
            sims = embs[:i] @ embs[i]
            novelty_vals.append(float(1.0 - np.max(sims)))
    except Exception:
        res.note("sentence-transformers unavailable, using lexical novelty fallback")
        seen = set()
        novelty_vals = []
        for ch in chunks:
            toks = set(_norm_token(t) for t in ch.split())
            toks.discard("")
            new = toks - seen
            novelty_vals.append(len(new) / max(1, len(toks)))
            seen |= toks

    from ..utils import resample_to_grid
    series = resample_to_grid(np.asarray(novelty_vals), np.asarray(chunk_times),
                              grid, fill=0.3, agg="mean")
    # forward-fill gaps with neutral novelty
    return smooth(np.clip(series, 0, 1), 3)


_ST_CACHE = {}


def _get_st_model(name: str = "all-MiniLM-L6-v2"):
    if name not in _ST_CACHE:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _ST_CACHE[name] = SentenceTransformer(name)
    return _ST_CACHE[name]
