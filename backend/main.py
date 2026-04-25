"""
main.py — FastAPI application for the Traffic Analyzer.

Routes
------
POST /upload          — Accept an MP4, queue a background processing job,
                        return a job_id.
GET  /status/{job_id} — Poll job status and progress percentage.
GET  /result/{job_id} — Retrieve full results once the job is complete.
GET  /video/{job_id}  — Stream the annotated output video file.
GET  /report/{job_id} — Download the CSV or Excel counting report.
WS   /ws/{job_id}     — WebSocket that streams status/progress until done.

State
-----
All job state lives in the module-level `jobs` dict (in-memory).  Contents
are lost on server restart — acceptable for single-session local deployment.
For multi-user or persistent use a database-backed store would be needed.

Static files (processed videos, reports) are served from ./outputs via the
/static mount so the frontend can load them directly by URL.
"""
import logging
import uuid
import time
import traceback
import asyncio
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from models import UploadResponse, StatusResponse, ResultResponse
from processor import process_video
from reporter import generate_csv, generate_excel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Traffic Analyzer")

app.add_middleware(
    CORSMiddleware,
    # Vite's default dev-server port.  Add production origins here if deploying.
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve processed videos and reports directly from disk.  The frontend
# constructs URLs like /static/<job_id>_processed.mp4 to display results.
app.mount("/static", StaticFiles(directory="outputs"), name="static")


@app.middleware("http")
async def add_static_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Cache-Control"] = "no-cache"
    return response

# In-memory job store: maps job_id → dict containing status, progress,
# file paths, and the full summary once processing completes.  Lost on
# server restart — acceptable for local single-session use.
jobs: dict = {}


def run_processing(job_id: str):
    """
    Background task that drives video processing for a single job.

    Triggered by FastAPI's BackgroundTasks immediately after /upload saves the
    file.  Updates jobs[job_id] throughout so /status and the WebSocket can
    report live progress to the frontend.

    On success: sets status="complete", writes summary, csv_path, excel_path.
    On failure: sets status="error" and records the exception string in "error".
    """
    if job_id not in jobs:
        logger.error(f"Job {job_id} missing from store before processing start")
        return

    job = jobs[job_id]
    job["status"] = "processing"
    # Recorded before process_video so the final duration includes model
    # initialisation and VideoWriter setup, not just the frame loop itself.
    start_time = time.time()

    # Closure so progress_callback can write directly into the job dict that
    # websocket_progress is reading; no queue or shared variable needed.
    def progress_callback(pct):
        job["progress"] = float(pct)

    try:
        result = process_video(
            video_path=job["video_path"],
            job_id=job_id,
            reid_mode=job["reid_mode"],
            line_mode=job["line_mode"],
            line_single_pos=job["line_single_pos"],
            line_dual_a_pos=job["line_dual_a_pos"],
            line_dual_b_pos=job["line_dual_b_pos"],
            progress_callback=progress_callback,
        )
        result["processing_duration_sec"] = round(time.time() - start_time, 3)

        vehicle_log = result.get("vehicle_log", [])
        job["csv_path"] = generate_csv(vehicle_log, result, job_id)
        job["excel_path"] = generate_excel(vehicle_log, result, job_id)
        job["summary"] = result
        job["status"] = "complete"
        job["progress"] = 100.0
        job["error"] = None

        logger.info(f"Job {job_id} complete")
    except Exception as e:
        # Catch-all so any failure in the CV pipeline is recorded in the job
        # dict rather than silently killing the background thread.  The frontend
        # learns about the error via the WebSocket status field, not an HTTP error.
        job["status"] = "error"
        job["error"] = str(e)
        logger.error(f"Job {job_id} failed: {e}")
        logger.error(traceback.format_exc())


@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    reid_mode: str = Form("fast"),
    line_mode: str = Form("dual"),
    line_single_pos: float = Form(0.50),
    line_dual_a_pos: float = Form(0.45),
    line_dual_b_pos: float = Form(0.80),
):
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted.")

    # job_id is generated before saving so the filename embeds the ID from the
    # start; every file on disk can be traced to its job without consulting
    # the in-memory dict, which helps with manual debugging.
    job_id = str(uuid.uuid4())[:8]
    dest = UPLOAD_DIR / f"{job_id}_{file.filename}"

    contents = await file.read()
    dest.write_bytes(contents)

    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "error": None,
        "summary": None,
        "csv_path": None,
        "excel_path": None,
        "reid_mode": reid_mode,
        "line_mode": line_mode,
        "line_single_pos": line_single_pos,
        "line_dual_a_pos": line_dual_a_pos,
        "line_dual_b_pos": line_dual_b_pos,
        "created_at": time.time(),
        "filename": file.filename,
        "video_path": str(dest),
    }
    # BackgroundTasks runs run_processing in a thread after the HTTP response
    # is sent, so the client receives the job_id immediately.  Using
    # asyncio.create_task would require run_processing to be a coroutine and
    # would block the event loop during the CPU-bound CV work.
    background_tasks.add_task(run_processing, job_id)
    logger.info(f"Job {job_id} queued")

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        reid_mode=reid_mode,
        line_mode=line_mode,
        line_single_pos=line_single_pos,
        line_dual_a_pos=line_dual_a_pos,
        line_dual_b_pos=line_dual_b_pos,
        status="pending",
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        error=job["error"],
    )


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "complete":
        # 425 "Too Early" signals the frontend that the result isn't ready
        # yet without throwing a hard error that would stop it from polling.
        return JSONResponse(
            status_code=425,
            content={
                "status": job["status"],
                "progress": job["progress"],
            },
        )

    summary = job.get("summary") or {}
    return ResultResponse(
        job_id=job_id,
        status=job["status"],
        total_unique=summary.get("total_unique", 0),
        count_by_class=summary.get("count_by_class", {}),
        processing_duration_sec=summary.get("processing_duration_sec", 0.0),
        reid_mode=summary.get("reid_mode", job["reid_mode"]),
        line_mode=summary.get("line_mode", job["line_mode"]),
        fps=summary.get("fps", 0.0),
        width=summary.get("width", 0),
        height=summary.get("height", 0),
        total_frames=summary.get("total_frames", 0),
        processed_frames=summary.get("processed_frames", 0),
        output_video_url=f"/static/{job_id}_processed.mp4",
        report_csv_url=f"/static/{job_id}_report.csv",
        report_excel_url=f"/static/{job_id}_report.xlsx",
    )


@app.get("/video/{job_id}")
async def get_video(job_id: str):
    job = jobs.get(job_id)
    if job is None or job["status"] != "complete":
        raise HTTPException(status_code=404, detail="Video not available")

    path = OUTPUT_DIR / f"{job_id}_processed.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Processed video file not found")

    return FileResponse(path, media_type="video/mp4")


@app.get("/report/{job_id}")
async def get_report(job_id: str, format: str = "csv"):
    job = jobs.get(job_id)
    if job is None or job["status"] != "complete":
        raise HTTPException(status_code=404, detail="Report not available")

    # format is a query param (?format=csv or ?format=excel) rather than a
    # path segment so both formats share one route pattern /report/{job_id}.
    report_format = format.lower()
    if report_format == "csv":
        csv_path = job.get("csv_path")
        # csv_path is None if the job completed without generating a report
        # (shouldn't happen in normal flow).  .exists() catches the case where
        # the outputs/ directory was cleared while the server was still running.
        if not csv_path or not Path(csv_path).exists():
            raise HTTPException(status_code=404, detail="CSV report not found")
        return FileResponse(
            csv_path,
            media_type="text/csv",
            filename=f"{job_id}_report.csv",
        )

    if report_format == "excel":
        excel_path = job.get("excel_path")
        if not excel_path or not Path(excel_path).exists():
            raise HTTPException(status_code=404, detail="Excel report not found")
        return FileResponse(
            excel_path,
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            filename=f"{job_id}_report.xlsx",
        )

    raise HTTPException(status_code=400, detail="Invalid format, use csv or excel")


@app.websocket("/ws/{job_id}")
async def websocket_progress(ws: WebSocket, job_id: str):
    await ws.accept()
    try:
        while True:
            if job_id not in jobs:
                await ws.send_json({"error": "job not found"})
                break

            job = jobs[job_id]
            await ws.send_json(
                {
                    "status": job["status"],
                    "progress": job["progress"],
                    "error": job["error"],
                }
            )

            # Break on error too, not just complete — continuing the loop after
            # an error would spam the same error message to the client every 500 ms
            # until it disconnects.  The frontend's onerror/onclose handler shows
            # the error state once the socket closes.
            if job["status"] in ("complete", "error"):
                break

            # Poll every 500 ms — fast enough for a smooth progress bar without
            # hammering the event loop or the in-memory job store.
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        # Client navigated away or closed the tab before processing finished;
        # this is normal browser behaviour and not a server-side error.
        logger.info(f"WebSocket disconnected: {job_id}")
