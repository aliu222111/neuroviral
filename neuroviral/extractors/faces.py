"""Face features: presence, relative size, expression intensity.

Tries mediapipe first, then an OpenCV Haar cascade, then degrades to "no faces"
(neutral). Expression intensity is proxied by frame-to-frame change inside the
detected face region (a real expression model can be dropped in later).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..utils import ExtractResult, TimeGrid, resample_to_grid, safe_norm, smooth

_MP_CACHE = {}


def _haar_cascade_path() -> str:
    """Path to the frontal-face Haar cascade.

    Prefers the copy bundled with the package (OpenCV 5.x no longer ships the
    cascade XMLs), falling back to ``cv2.data.haarcascades`` when present.
    """
    from pathlib import Path
    bundled = Path(__file__).with_name("data") / "haarcascade_frontalface_default.xml"
    if bundled.exists():
        return str(bundled)
    try:
        import cv2  # type: ignore
        return cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    except Exception:
        return str(bundled)


def _detect_mediapipe(frames) -> Optional[List[Optional[Tuple[float, float, float, float]]]]:
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None
    try:
        detector = _MP_CACHE.get("fd")
        if detector is None:
            detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
            _MP_CACHE["fd"] = detector
        boxes = []
        for f in frames:
            result = detector.process(f)
            if result.detections:
                d = max(result.detections,
                        key=lambda x: x.location_data.relative_bounding_box.width
                        * x.location_data.relative_bounding_box.height)
                bb = d.location_data.relative_bounding_box
                boxes.append((bb.xmin, bb.ymin, bb.width, bb.height))
            else:
                boxes.append(None)
        return boxes
    except Exception:
        return None


def _detect_haar(frames) -> Optional[List[Optional[Tuple[float, float, float, float]]]]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    try:
        cascade = cv2.CascadeClassifier(_haar_cascade_path())
        if cascade.empty():
            return None
        boxes = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2, 4)
            H, W = gray.shape[:2]
            if len(faces):
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                boxes.append((x / W, y / H, w / W, h / H))
            else:
                boxes.append(None)
        return boxes
    except Exception:
        return None


def extract(frames: List[np.ndarray], frame_fps: float, grid: TimeGrid) -> ExtractResult:
    res = ExtractResult()
    if not frames or frame_fps <= 0:
        res.series["face_presence"] = grid.zeros()
        res.series["face_size"] = grid.zeros()
        res.series["expression_intensity"] = grid.full(0.3)
        res.note("no frames — face features neutral", degraded=True)
        return res

    boxes = _detect_mediapipe(frames)
    method = "mediapipe"
    if boxes is None:
        boxes = _detect_haar(frames)
        method = "opencv-haar"
    if boxes is None:
        res.series["face_presence"] = grid.zeros()
        res.series["face_size"] = grid.zeros()
        res.series["expression_intensity"] = grid.full(0.3)
        res.note("no face detector available (mediapipe/opencv) — faces neutral",
                 degraded=True)
        return res

    times = np.arange(len(frames)) / frame_fps
    presence = np.asarray([1.0 if b is not None else 0.0 for b in boxes])
    size = np.asarray([(b[2] * b[3]) if b is not None else 0.0 for b in boxes])
    size = np.clip(size / 0.35, 0, 1)  # face covering ~35% of frame => full

    # Expression intensity: change inside the face box between frames.
    expr = np.zeros(len(frames))
    prev = None
    for i, (f, b) in enumerate(zip(frames, boxes)):
        if b is None:
            expr[i] = 0.0
            prev = None
            continue
        H, W = f.shape[:2]
        x0 = int(max(0, b[0]) * W); y0 = int(max(0, b[1]) * H)
        x1 = int(min(1, b[0] + b[2]) * W); y1 = int(min(1, b[1] + b[3]) * H)
        crop = f[y0:y1, x0:x1].astype(np.float32).mean(axis=-1)
        if prev is not None and prev.shape == crop.shape and crop.size:
            expr[i] = float(np.mean(np.abs(crop - prev)) / 255.0)
        prev = crop

    res.series["face_presence"] = smooth(resample_to_grid(presence, times, grid, agg="max"), 3)
    res.series["face_size"] = smooth(resample_to_grid(size, times, grid, agg="mean"), 3)
    res.series["expression_intensity"] = safe_norm(smooth(resample_to_grid(expr, times, grid, agg="mean"), 3))
    res.scalars["face_fraction"] = float(np.mean(presence))
    res.scalars["detector"] = method
    return res
