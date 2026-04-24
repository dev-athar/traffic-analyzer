import csv
from pathlib import Path

from openpyxl import Workbook


OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
REPORT_TITLE = "Smart Drone Traffic Analyzer Report"

REPORT_COLUMNS = [
    "frame_index",
    "timestamp_sec",
    "track_id",
    "vehicle_class",
    "counter",
]


def _autosize_worksheet_columns(worksheet):
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            cell_value = "" if cell.value is None else str(cell.value)
            if len(cell_value) > max_length:
                max_length = len(cell_value)
        worksheet.column_dimensions[column_letter].width = max_length + 2


def _build_counted_rows(vehicle_log: list) -> list:
    counted_rows = []
    seen_counted_ids = set()
    counter = 0

    for row in vehicle_log:
        track_id = row.get("track_id")
        if not row.get("counted"):
            continue
        if track_id in seen_counted_ids:
            continue

        seen_counted_ids.add(track_id)
        counter += 1
        counted_rows.append(
            {
                "frame_index": row.get("frame_index"),
                "timestamp_sec": row.get("timestamp_sec"),
                "track_id": track_id,
                "vehicle_class": row.get("vehicle_class"),
                "counter": counter,
            }
        )

    return counted_rows


def generate_csv(vehicle_log: list, summary: dict, job_id: str) -> str:
    output_path = OUTPUT_DIR / f"{job_id}_report.csv"
    counted_rows = _build_counted_rows(vehicle_log)
    processing_duration = summary.get("processing_duration_sec", 0.0)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([REPORT_TITLE])
        writer.writerow(["Job ID", job_id])
        writer.writerow(["Processing duration (sec)", processing_duration])
        writer.writerow([])
        writer.writerow(REPORT_COLUMNS)
        for row in counted_rows:
            writer.writerow([row.get(column) for column in REPORT_COLUMNS])

    return str(output_path)


def generate_excel(vehicle_log: list, summary: dict, job_id: str) -> str:
    output_path = OUTPUT_DIR / f"{job_id}_report.xlsx"
    workbook = Workbook()
    counted_rows = _build_counted_rows(vehicle_log)

    # Sheet 1: Counted vehicles only
    detections_sheet = workbook.active
    detections_sheet.title = "Counted"
    detections_sheet.append([REPORT_TITLE])
    detections_sheet.append(["Job ID", job_id])
    detections_sheet.append(
        ["Processing duration (sec)", summary.get("processing_duration_sec", 0.0)]
    )
    detections_sheet.append([])
    detections_sheet.append(REPORT_COLUMNS)
    for row in counted_rows:
        detections_sheet.append([row.get(column) for column in REPORT_COLUMNS])
    _autosize_worksheet_columns(detections_sheet)

    # Sheet 2: Summary
    summary_sheet = workbook.create_sheet("Summary")
    summary_sheet.append(["Metric", "Value"])
    summary_sheet.append(["Total unique vehicles", summary.get("total_unique", 0)])

    count_by_class = summary.get("count_by_class", {}) or {}
    if count_by_class:
        for vehicle_class, class_count in count_by_class.items():
            summary_sheet.append([f"Count by class: {vehicle_class}", class_count])
    else:
        summary_sheet.append(["Count by class", "{}"])

    summary_sheet.append(
        ["Processing duration (sec)", summary.get("processing_duration_sec", 0.0)]
    )
    summary_sheet.append(["ReID mode", summary.get("reid_mode", "")])
    summary_sheet.append(["Line mode", summary.get("line_mode", "")])
    summary_sheet.append(["Video FPS", summary.get("fps", 0.0)])

    width = summary.get("width", 0)
    height = summary.get("height", 0)
    summary_sheet.append(["Video resolution", f"{width}x{height}"])
    summary_sheet.append(["Total frames", summary.get("total_frames", 0)])
    summary_sheet.append(["Processed frames", summary.get("processed_frames", 0)])

    _autosize_worksheet_columns(summary_sheet)
    workbook.save(output_path)

    return str(output_path)
