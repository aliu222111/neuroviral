"""FastAPI dashboard: upload a clip, get scores + a synced brain timeline.

Run:  uvicorn dashboard.server:app --reload
  or:  python -m dashboard.server

Endpoints
    GET  /               -> single-page app
    POST /analyze        -> multipart upload; returns {id, result}
    GET  /result/{id}    -> full JSON result
    GET  /video/{id}     -> the uploaded clip (for the player)
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Dict

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI is required for the dashboard: pip install -e '.[web]'"
    ) from exc

from neuroviral.pipeline import analyze

app = FastAPI(title="NeuroViral", version="0.1.0")

_STATIC = Path(__file__).parent / "static"
_STORE_DIR = Path(tempfile.gettempdir()) / "neuroviral_uploads"
_STORE_DIR.mkdir(exist_ok=True)
_RESULTS: Dict[str, dict] = {}
_VIDEOS: Dict[str, str] = {}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_file = _STATIC / "index.html"
    if index_file.exists():
        return index_file.read_text()
    return "<h1>NeuroViral</h1><p>static app missing</p>"


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    platform: str = "both",
    content_type: str = "auto",
):
    vid = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    dest = _STORE_DIR / f"{vid}{suffix}"
    dest.write_bytes(await file.read())
    _VIDEOS[vid] = str(dest)

    platforms = ["tiktok", "reels"] if platform == "both" else [platform]
    try:
        result = analyze(str(dest), platforms=platforms, content_type=content_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"analysis failed: {exc}")

    payload = result.to_dict()
    payload["id"] = vid
    _RESULTS[vid] = payload
    return JSONResponse(payload)


@app.get("/result/{vid}")
def get_result(vid: str):
    if vid not in _RESULTS:
        raise HTTPException(status_code=404, detail="unknown id")
    return JSONResponse(_RESULTS[vid])


@app.get("/video/{vid}")
def get_video(vid: str):
    if vid not in _VIDEOS:
        raise HTTPException(status_code=404, detail="unknown id")
    return FileResponse(_VIDEOS[vid])


def main():  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
