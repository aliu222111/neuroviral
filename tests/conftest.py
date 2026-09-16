"""Shared test fixtures + synthetic clip generation (no committed binaries).

Clips are generated at test time with ffmpeg. Tests that need ffmpeg are skipped
automatically when it is unavailable, so the unit suite still runs green.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")
has_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def make_high_energy_clip(path: str, dur: float = 4.0):
    """Fast-cut, high-motion, loud clip (strong hook)."""
    # Rapidly changing testsrc + audible tone => motion + energy from t=0.
    vf = "format=rgb24"
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={dur}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
        "-vf", vf, "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(path),
    ]
    _run(cmd)


def make_slow_start_clip(path: str, dur: float = 4.0, dead: float = 2.5):
    """Static, silent opening (dead air) then a little motion — weak hook."""
    # First `dead` seconds: black + silence. Then testsrc + tone.
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", f"color=c=black:size=320x240:rate=30:duration={dead}",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={dur-dead}",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono:duration={dead}",
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={dur-dead}",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v];[2:a][3:a]concat=n=2:v=0:a=1[a]",
        "-map", "[v]", "-map", "[a]", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(path),
    ]
    _run(cmd)


def make_silent_short_clip(path: str, dur: float = 0.6):
    """Very short, no audio track — edge case."""
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=30:duration={dur}",
        "-an", "-pix_fmt", "yuv420p", str(path),
    ]
    _run(cmd)


@pytest.fixture(scope="session")
def clips(tmp_path_factory):
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("clips")
    high = d / "high_energy.mp4"
    slow = d / "slow_start.mp4"
    short = d / "short_silent.mp4"
    make_high_energy_clip(str(high))
    make_slow_start_clip(str(slow))
    make_silent_short_clip(str(short))
    return {"high": str(high), "slow": str(slow), "short": str(short)}
