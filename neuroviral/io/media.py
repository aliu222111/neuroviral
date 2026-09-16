"""ffmpeg-backed media decoding: probe metadata, extract audio, sample frames.

Uses the system `ffmpeg`/`ffprobe` binaries directly (no python bindings
required) so the only hard dependency here is a working ffmpeg install plus
numpy. Everything degrades gracefully: a video with no audio track yields
``has_audio=False`` and a ``None`` waveform rather than a crash.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


class MediaError(RuntimeError):
    pass


def _find(binary: str) -> Optional[str]:
    return shutil.which(binary)


def ffmpeg_available() -> bool:
    return _find("ffmpeg") is not None and _find("ffprobe") is not None


@dataclass
class MediaInfo:
    path: str
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    has_audio: bool = False
    has_video: bool = False
    notes: List[str] = field(default_factory=list)


def probe(path: str) -> MediaInfo:
    """Probe duration / fps / resolution / stream presence via ffprobe."""
    info = MediaInfo(path=str(path))
    p = Path(path)
    if not p.exists():
        raise MediaError(f"file not found: {path}")
    ffprobe = _find("ffprobe")
    if ffprobe is None:
        info.notes.append("ffprobe not found; metadata unavailable")
        return info
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120)
        data = json.loads(out.stdout.decode("utf-8", "ignore") or "{}")
    except Exception as exc:  # pragma: no cover - defensive
        info.notes.append(f"ffprobe failed: {exc}")
        return info

    fmt = data.get("format", {})
    try:
        info.duration = float(fmt.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0

    for st in data.get("streams", []):
        codec_type = st.get("codec_type")
        if codec_type == "video" and not info.has_video:
            info.has_video = True
            info.width = int(st.get("width", 0) or 0)
            info.height = int(st.get("height", 0) or 0)
            info.fps = _parse_fps(st.get("avg_frame_rate") or st.get("r_frame_rate"))
            if info.duration <= 0:
                info.duration = _stream_duration(st)
        elif codec_type == "audio":
            info.has_audio = True

    if info.duration <= 0:
        info.notes.append("could not determine duration")
    return info


def _parse_fps(rate: Optional[str]) -> float:
    if not rate or rate == "0/0":
        return 0.0
    try:
        if "/" in rate:
            num, den = rate.split("/")
            den = float(den)
            return float(num) / den if den else 0.0
        return float(rate)
    except (TypeError, ValueError):
        return 0.0


def _stream_duration(st: dict) -> float:
    try:
        return float(st.get("duration", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class AudioWaveform:
    samples: np.ndarray  # mono float32 in [-1, 1]
    sr: int


def extract_audio(path: str, sr: int = 16000) -> Optional[AudioWaveform]:
    """Decode the audio track to mono PCM at ``sr`` Hz.

    Returns ``None`` if there is no audio track or ffmpeg is unavailable.
    """
    ffmpeg = _find("ffmpeg")
    if ffmpeg is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    cmd = [
        ffmpeg, "-y", "-i", str(path), "-vn",
        "-ac", "1", "-ar", str(sr), "-f", "wav", wav_path,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=600)
        if res.returncode != 0 or not Path(wav_path).exists():
            return None
        with wave.open(wav_path, "rb") as wf:
            n = wf.getnframes()
            if n == 0:
                return None
            raw = wf.readframes(n)
            width = wf.getsampwidth()
            channels = wf.getnchannels()
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
        data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        maxval = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
        data = data / max(1.0, maxval)
        if data.size == 0:
            return None
        return AudioWaveform(samples=data, sr=sr)
    except Exception:
        return None
    finally:
        try:
            Path(wav_path).unlink(missing_ok=True)
        except Exception:  # pragma: no cover
            pass


def sample_frames(path: str, fps: float = 2.0, max_width: int = 320) -> List[np.ndarray]:
    """Sample frames at ``fps``, downscaled, as a list of HxWx3 uint8 arrays.

    Uses ffmpeg's rawvideo rgb24 pipe. Returns an empty list on any failure
    (callers must degrade gracefully).
    """
    ffmpeg = _find("ffmpeg")
    if ffmpeg is None:
        return []
    info = probe(path)
    if not info.has_video or info.width <= 0 or info.height <= 0:
        return []
    scale = min(1.0, max_width / info.width)
    w = max(2, int(round(info.width * scale)) // 2 * 2)
    h = max(2, int(round(info.height * scale)) // 2 * 2)
    cmd = [
        ffmpeg, "-v", "quiet", "-i", str(path),
        "-vf", f"fps={fps},scale={w}:{h}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=600)
        buf = res.stdout
        frame_bytes = w * h * 3
        if frame_bytes == 0 or len(buf) < frame_bytes:
            return []
        n_frames = len(buf) // frame_bytes
        arr = np.frombuffer(buf[: n_frames * frame_bytes], dtype=np.uint8)
        return list(arr.reshape(n_frames, h, w, 3))
    except Exception:
        return []
