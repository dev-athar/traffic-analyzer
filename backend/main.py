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

app = FastAPI(title="Smart Drone Traffic Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="outputs"), name="static")


@app.middleware("http")
async def add_static_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Accept-Ranges"] = "bytes"
        response.headers["Cache-Control"] = "no-cache"
    return response

jobs: dict = {}


def run_processing(job_id: str):
    if job_id not in jobs:
        logger.error(f"Job {job_id} missing from store before processing start")
        return

    job = jobs[job_id]
    job["status"] = "processing"
    start_time = time.time()

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

    report_format = format.lower()
    if report_format == "csv":
        csv_path = job.get("csv_path")
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

            if job["status"] in ("complete", "error"):
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {job_id}")
