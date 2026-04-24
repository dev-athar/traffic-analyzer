"""
processor.py — Video processing pipeline orchestrator.

Opens a video file, drives the frame loop, delegates detection and tracking
to VehicleTracker, writes an annotated output video, and returns a summary
dict. Exposes process_video() for use by main.py endpoints and directly
as a CLI script for standalone testing.
"""

import logging
import os
import cv2

from tracker import VehicleTracker, FRAME_SKIP

logger = logging.getLogger(__name__)


def process_video(
    video_path: str,
    job_id: str,
    reid_mode: str = "fast",
    line_mode: str = "dual",
    line_single_pos: float = 0.50,
    line_dual_a_pos: float = .50,
    line_dual_b_pos: float = 0.80,
    progress_callback=None,
) -> dict:
    # 1. Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        f"Job {job_id}: total_frames={total_frames}, fps={fps}, "
        f"width={width}, height={height}"
    )

    # 2. Output video writer
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/{job_id}_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 3. Tracker
    tracker = VehicleTracker(
        reid_mode=reid_mode,
        line_mode=line_mode,
        line_single_pos=line_single_pos,
        line_dual_a_pos=line_dual_a_pos,
        line_dual_b_pos=line_dual_b_pos,
    )
    tracker.set_frame_dimensions(width, height)

    # 4. Frame loop
    frame_index     = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % FRAME_SKIP == 0:
            annotated, _ = tracker.process_frame(frame, frame_index, fps)
            out.write(annotated)
            processed_count += 1
        else:
            out.write(frame)

        frame_index += 1

        if progress_callback and frame_index % 10 == 0:
            pct = (frame_index / total_frames) * 100 if total_frames else 0
            progress_callback(round(pct, 1))

    # 5. Release resources
    cap.release()
    out.release()

    # 6. Build summary
    summary = tracker.get_summary()
    summary.update({
        "job_id":           job_id,
        "video_path":       video_path,
        "output_path":      output_path,
        "fps":              fps,
        "width":            width,
        "height":           height,
        "total_frames":     total_frames,
        "processed_frames": processed_count,
        "reid_mode":        reid_mode,
        "line_mode":        line_mode,
        "line_single_pos":  line_single_pos,
        "line_dual_a_pos":  line_dual_a_pos,
        "line_dual_b_pos":  line_dual_b_pos,
    })

    logger.info(
        f"Job {job_id} complete. Total counted: {summary['total_unique']}"
    )

    return summary


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    if len(sys.argv) < 2:
        print(
            "Usage: python processor.py <video.mp4> "
            "[fast|accurate] [single|dual] [line_pos]\n"
            # fast     = YOLOv8n, 640px  (faster)
            # accurate = YOLOv8s, 1280px (better detection)
        )
        sys.exit(1)

    video_path      = sys.argv[1]
    job_id          = "test_job_001"
    reid_mode       = sys.argv[2] if len(sys.argv) > 2 else "fast"
    line_mode       = sys.argv[3] if len(sys.argv) > 3 else "single"
    line_single_pos = float(sys.argv[4]) if len(sys.argv) > 4 else 0.50

    print(f"Starting test run...")
    print(f"Video: {video_path}")
    print(f"ReID mode: {reid_mode}")
    print(f"Line mode: {line_mode}")
    print("-" * 40)

    def print_progress(pct):
        print(f"Progress: {pct:.1f}%", end="\r", flush=True)

    summary = process_video(
        video_path=video_path,
        job_id=job_id,
        reid_mode=reid_mode,
        line_mode=line_mode,
        line_single_pos=line_single_pos,
        progress_callback=print_progress,
    )

    print("\n" + "=" * 40)
    print("PROCESSING COMPLETE")
    print("=" * 40)
    print(f"Total unique vehicles counted : {summary['total_unique']}")
    print("By class:")
    for cls, count in summary["count_by_class"].items():
        print(f"  {cls:<12} : {count}")
    print(f"Processed frames : {summary['processed_frames']} / {summary['total_frames']}")
    print(f"Output video     : {summary['output_path']}")
    print("=" * 40)
