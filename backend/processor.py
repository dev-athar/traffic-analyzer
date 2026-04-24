"""
processor.py — Outer pipeline loop for the drone traffic analyzer.

This module is the bridge between the FastAPI layer (main.py) and the tracking
core (tracker.py).  process_video() performs these steps in order:

  1. Open the source video with OpenCV and read its metadata (fps, dimensions).
  2. Create an H.264 VideoWriter for the annotated output file.
  3. Instantiate VehicleTracker with the caller-supplied parameters.
  4. Drive the frame loop:
       - Every Nth frame (controlled by FRAME_SKIP) is passed to the tracker
         and the annotated result is written to the output.
       - Skipped frames are written as-is so output video timing stays in sync.
  5. Release all handles, merge the tracker summary with job metadata, return.

Exposed as a standalone CLI script for quick testing (see __main__ block).
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
    """
    Process a video file end-to-end and return a counting summary dict.

    Parameters
    ----------
    video_path       : absolute path to the source .mp4 file.
    job_id           : unique job identifier; used to name output files.
    reid_mode        : "fast" (YOLOv8n) or "accurate" (YOLOv8s, 1280px).
    line_mode        : "single" or "dual" counting-line layout.
    line_single_pos  : fractional frame height of the single line (0–1).
    line_dual_a_pos  : fractional frame height of Line A in dual mode.
    line_dual_b_pos  : fractional frame height of Line B in dual mode.
    progress_callback: optional callable(float) invoked with 0–100 progress.

    Returns
    -------
    dict — VehicleTracker summary augmented with job metadata (job_id, fps,
           dimensions, frame counts, output file paths, configuration values).

    Raises
    ------
    ValueError — if the video file cannot be opened by OpenCV.
    """
    # 1. Open video
    cap = cv2.VideoCapture(video_path)
    # cap.isOpened() is False for missing files or unsupported codecs; raising
    # here avoids a silent failure deep inside the frame loop.
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
    # avc1 = H.264 — produces MP4s that browsers can play inline without
    # transcoding.  Use "mp4v" as a fallback if avc1 is unavailable on Linux.
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
        # ret is False at end-of-file or on a corrupted frame; exit cleanly.
        ret, frame = cap.read()
        if not ret:
            break

        # Run the tracker on every Nth frame to control CPU load.
        if frame_index % FRAME_SKIP == 0:
            annotated, _ = tracker.process_frame(frame, frame_index, fps)
            out.write(annotated)
            processed_count += 1
        else:
            # Write the original frame so the output video frame count and
            # playback timing stay identical to the source (no speed-up glitch).
            out.write(frame)

        frame_index += 1

        # Fire the callback every 10 frames — frequent enough for a smooth
        # progress bar but not so frequent it adds measurable overhead.
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
