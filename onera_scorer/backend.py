"""FastAPI application routes for Onera SleepScorer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .session import SessionManager

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Onera SleepScorer", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

manager = SessionManager()


class ScanRequest(BaseModel):
    data_dir: str
    output_dir: str = ""


class SelectRecordingRequest(BaseModel):
    recording_key: str


class StageEditRequest(BaseModel):
    epochs: list[int]
    stage: str


class MaEditRequest(BaseModel):
    epochs: list[int]


class ConsensusAcceptRequest(BaseModel):
    epochs: list[int]
    scorer: int


class EnsureChannelRequest(BaseModel):
    channel: str


class ExportRequest(BaseModel):
    fmt_key: str = "markers"
    output_dir: str = "."
    filename: str = ""
    use_rk: bool = False
    include_events: bool = True


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return manager.get_initial_state()


@app.post("/api/scan")
def scan_dirs(req: ScanRequest) -> dict[str, Any]:
    try:
        return manager.scan_directories(req.data_dir, req.output_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/recording/select")
def select_recording(req: SelectRecordingRequest) -> dict[str, Any]:
    try:
        return manager.select_recording(req.recording_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/score/start")
def start_score(
    config: str = Form(...),
    resume_file: UploadFile | None = File(None),
    scorer1_file: UploadFile | None = File(None),
    scorer2_file: UploadFile | None = File(None),
) -> dict[str, str]:
    try:
        cfg = json.loads(config)
        if not isinstance(cfg, dict):
            raise ValueError("Invalid scoring config payload.")
        resume_bytes: bytes | None = None
        resume_name: str | None = None
        if resume_file is not None and resume_file.filename:
            resume_bytes = resume_file.file.read()
            resume_name = resume_file.filename
        s1_bytes: bytes | None = None
        s1_name: str | None = None
        if scorer1_file is not None and scorer1_file.filename:
            s1_bytes = scorer1_file.file.read()
            s1_name = scorer1_file.filename
        s2_bytes: bytes | None = None
        s2_name: str | None = None
        if scorer2_file is not None and scorer2_file.filename:
            s2_bytes = scorer2_file.file.read()
            s2_name = scorer2_file.filename
        job_id = manager.start_scoring_job(
            cfg,
            resume_bytes=resume_bytes,
            resume_filename=resume_name,
            scorer1_bytes=s1_bytes,
            scorer1_filename=s1_name,
            scorer2_bytes=s2_bytes,
            scorer2_filename=s2_name,
        )
        return {"job_id": job_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/score/status/{job_id}")
def get_score_status(job_id: str) -> dict[str, Any]:
    job = manager.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.get("/api/score/progress/{job_id}")
async def stream_score_progress(job_id: str) -> StreamingResponse:
    async def _event_stream():
        last_sig = None
        while True:
            job = manager.get_job_status(job_id)
            if job is None:
                payload = json.dumps({"status": "error", "error": f"Unknown job: {job_id}"})
                yield f"data: {payload}\n\n"
                return
            sig = (job.get("status"), job.get("pct"), job.get("subtitle"), job.get("error"))
            if sig != last_sig:
                last_sig = sig
                yield f"data: {json.dumps(job)}\n\n"
            if job.get("status") in ("done", "error"):
                return
            await asyncio.sleep(0.1)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/session")
def get_session() -> dict[str, Any]:
    try:
        return manager.get_serialized_session()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/edits/stage")
def edit_stage(req: StageEditRequest) -> dict[str, Any]:
    try:
        return manager.apply_stage_edit(req.epochs, req.stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/edits/ma")
def edit_ma(req: MaEditRequest) -> dict[str, Any]:
    try:
        return manager.toggle_ma(req.epochs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/edits/consensus_accept")
def edit_consensus_accept(req: ConsensusAcceptRequest) -> dict[str, Any]:
    try:
        return manager.accept_consensus_scorer(req.epochs, req.scorer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/edits/undo")
def edit_undo() -> dict[str, Any]:
    try:
        return manager.undo_last_edit()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/edits/clear")
def edit_clear() -> dict[str, Any]:
    try:
        return manager.clear_all_edits()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/channels/ensure")
def ensure_channel(req: EnsureChannelRequest) -> dict[str, Any]:
    res = manager.ensure_display_channel(req.channel)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Channel unavailable: {req.channel}")
    return {
        "name": res["name"],
        "type": res["type"],
        "unit": res["unit"],
        "color": res["color"],
        "ylim": res["ylim"],
        "encodeMax": res["encodeMax"],
    }


@app.get("/api/waveforms")
def get_waveforms(
    start_ep: int = Query(..., ge=0),
    end_ep: int = Query(..., gt=0),
    channels: list[str] = Query(...),
) -> Response:
    try:
        with manager._lock:
            known_chs = set(
                (manager.scored_bundle.get("all_recording_chs") or [])
                if manager.scored_bundle is not None
                else []
            )
        expanded: list[str] = []
        for item in channels:
            if item in known_chs or "," not in item:
                expanded.append(item)
            else:
                expanded.extend(part for part in item.split(",") if part)
        raw_bytes = manager.build_waveforms_binary(start_ep, end_ep, expanded)
        return Response(content=raw_bytes, media_type="application/octet-stream")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/export/preview")
def export_preview(req: ExportRequest) -> dict[str, Any]:
    try:
        return manager.preview_export(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export/save")
def export_save(req: ExportRequest) -> dict[str, Any]:
    try:
        return manager.save_export(req.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export/download")
def export_download(req: ExportRequest) -> Response:
    try:
        raw_bytes, fname = manager.download_export(req.model_dump())
        safe_name = fname.replace('"', "")
        return Response(
            content=raw_bytes,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
