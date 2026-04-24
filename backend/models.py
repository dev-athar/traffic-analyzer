from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    reid_mode: str
    line_mode: str
    line_single_pos: float
    line_dual_a_pos: float
    line_dual_b_pos: float
    status: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    error: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    status: str
    total_unique: int
    count_by_class: dict
    processing_duration_sec: float
    reid_mode: str
    line_mode: str
    fps: float
    width: int
    height: int
    total_frames: int
    processed_frames: int
    output_video_url: str
    report_csv_url: str
    report_excel_url: str
