"""
tracker.py — Core tracking and counting module for the drone traffic analyzer.

VehicleTracker wraps YOLOv8 (via Ultralytics) and ByteTrack to detect and
persistently track vehicles across video frames.  It handles:

  Model selection
    YOLOv8n (fast mode) or YOLOv8s (accurate mode, 1280px inference).

  Line-crossing counting
    Single-line or dual-line layout, each line positioned by a fractional
    frame-height value.  A vehicle is counted the first time its centroid
    crosses a line; crossed_ids prevents it being counted again on the same
    track ID.

  ReID duplicate suppression
    When a re-entering vehicle gets a new ByteTrack ID after occlusion,
    _reid_check() compares its HSV histogram against recently counted entries.
    If the correlation score exceeds REID_THRESHOLD the count is suppressed.

  Flash effect
    The bbox is filled semi-transparently for FLASH_FRAMES frames after a
    confirmed crossing, giving clear visual feedback in the output video.
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO
logger = logging.getLogger(__name__)

# --- Tunable constants ---

# Default counting-line layout used when no override is passed by the caller.
# "single" = one line across the frame; "dual" = two lines (A and B).
LINE_MODE        = "single"

# Vertical position of the single counting line as a fraction of frame height.
# 0.50 = middle of the frame.  Move toward 0.0 for top, 1.0 for bottom.
LINE_SINGLE_POS  = 0.50

# Top line position in dual mode (fraction of frame height).
# Lower this value to move Line A toward the top of the frame.
LINE_DUAL_A_POS  = 0.25

# Bottom line position in dual mode (fraction of frame height).
# Raise this value to move Line B toward the bottom of the frame.
LINE_DUAL_B_POS  = 0.75

# Number of frames the semi-transparent bbox fill is shown after a crossing.
# Increase for a longer flash; decrease to shorten it.
FLASH_FRAMES     = 10

# How many frames ByteTrack keeps a lost track alive after the vehicle
# disappears.  Increase for scenes with long occlusions; decrease to free
# memory sooner in dense traffic.
MAX_TRACK_AGE    = 50

# Histogram correlation threshold for ReID duplicate suppression (0.0–1.0).
# A score above this means "same vehicle — skip count".  Raise it to be
# stricter (allow more re-counts); lower it to suppress more aggressively.
REID_THRESHOLD   = 0.90

# Process every Nth frame.  1 = no skipping (every frame is tracked).
# Increase (e.g. 2 or 3) to trade counting accuracy for speed on slow hardware.
FRAME_SKIP       = 1


class VehicleTracker:
    def __init__(
        self,
        reid_mode: str = "fast",
        line_mode: str = "single",
        line_single_pos: float = 0.50,
        line_dual_a_pos: float = 0.25,
        line_dual_b_pos: float = 0.75,
    ):
        """
        Initialise the tracker, load the YOLO model, and set up all state stores.

        Parameters
        ----------
        reid_mode        : "fast" loads YOLOv8n at default resolution;
                           "accurate" loads YOLOv8s with 1280px inference.
        line_mode        : "single" for one counting line; "dual" for two lines.
        line_single_pos  : fractional frame height for the single line (0–1).
        line_dual_a_pos  : fractional frame height for Line A in dual mode.
        line_dual_b_pos  : fractional frame height for Line B in dual mode.
        """
        self.reid_mode       = reid_mode
        self.line_mode       = line_mode
        self.line_single_pos = line_single_pos
        self.line_dual_a_pos = line_dual_a_pos
        self.line_dual_b_pos = line_dual_b_pos

        if self.reid_mode == "accurate":
            self.model = YOLO("yolov8s.pt")
            logger.info("Model: yolov8s.pt (accurate)")
        else:
            self.model = YOLO("yolov8n.pt")
            logger.info("Model: yolov8n.pt (fast)")

        self.crossed_ids      = set()
        self.track_last_side  = {}
        self.recently_counted = []
        self.vehicle_log      = []
        self.count_by_class   = {}
        self.flash_registry   = {}

        self.line_y_a = None
        self.line_y_b = None

        logger.info(
            f"VehicleTracker initialized, "
            f"reid_mode={reid_mode}, "
            f"line_mode={line_mode}"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_frame_dimensions(self, width: int, height: int):
        """
        Convert fractional line positions to absolute pixel y-coordinates.

        Must be called once after the video is opened and before any frames
        are processed.  Stores frame_width and frame_height for use in drawing.

        Parameters
        ----------
        width  : frame width in pixels.
        height : frame height in pixels.
        """
        if self.line_mode == "single":
            self.line_y_a = int(height * self.line_single_pos)
            self.line_y_b = None
        else:
            self.line_y_a = int(height * self.line_dual_a_pos)
            self.line_y_b = int(height * self.line_dual_b_pos)
        self.frame_height = height
        self.frame_width  = width
        logger.info(
            f"Frame {width}x{height}, "
            f"line_a={self.line_y_a}, "
            f"line_b={self.line_y_b}"
        )

    # ------------------------------------------------------------------
    # Per-frame entry point
    # ------------------------------------------------------------------

    def process_frame(self, frame, frame_index: int, fps: float):
        """
        Run detection and tracking on one video frame; return annotated result.

        In accurate mode the frame is downscaled to 1280px width for inference
        and bboxes are scaled back to original pixel coordinates before any
        crossing logic runs — all downstream code always works in original pixels.

        Parameters
        ----------
        frame       : np.ndarray — BGR frame from the video capture.
        frame_index : int        — zero-based index of the current frame.
        fps         : float      — video frame rate used to compute timestamps.

        Returns
        -------
        annotated  : np.ndarray — frame with bboxes, labels, and lines drawn.
        detections : list       — list of detection dicts for this frame.
        """
        if self.reid_mode == "accurate":
            original_h, original_w = frame.shape[:2]
            scale = min(1.0, 1280 / original_w)
            infer_w = int(original_w * scale)
            infer_h = int(original_h * scale)
            infer_frame = cv2.resize(frame, (infer_w, infer_h))
            results = self.model.track(
                infer_frame,
                persist=True,
                classes=[2, 3, 5, 7],
                verbose=False,
                tracker="bytetrack.yaml",
                imgsz=1280,
                conf=0.35,
            )
            # Ratios to map inference-space coordinates back to original pixels.
            scale_x = original_w / infer_w
            scale_y = original_h / infer_h
        else:
            results = self.model.track(
                frame,
                persist=True,
                classes=[2, 3, 5, 7],
                verbose=False,
                tracker="bytetrack.yaml",
            )

        if results is None or results[0].boxes is None:
            return frame, []

        boxes = results[0].boxes
        detections = []

        cls_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

        for box in boxes:
            if box.id is None:
                continue
            track_id   = int(box.id.item())
            cls_id     = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if self.reid_mode == "accurate":
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

            cls_name = cls_map.get(cls_id, "unknown")

            crossed, counted = self._check_line_crossing(
                track_id, cls_name, confidence,
                [x1, y1, x2, y2], frame,
            )

            detections.append({
                "track_id":   track_id,
                "cls":        cls_name,
                "confidence": confidence,
                "bbox":       [x1, y1, x2, y2],
                "crossed":    crossed,
                "counted":    counted,
            })

            self.vehicle_log.append({
                "frame_index":   frame_index,
                "timestamp_sec": round(frame_index / fps, 3),
                "track_id":      track_id,
                "vehicle_class": cls_name,
                "confidence":    round(confidence, 3),
                "bbox_x1": x1, "bbox_y1": y1,
                "bbox_x2": x2, "bbox_y2": y2,
                "crossed_line":  crossed,
                "counted":       counted,
            })

        annotated = self._draw_frame(frame.copy(), detections)
        return annotated, detections

    # ------------------------------------------------------------------
    # Line-crossing logic
    # ------------------------------------------------------------------

    def _check_line_crossing(
        self,
        track_id: int,
        cls_name: str,
        confidence: float,
        bbox: list,
        frame,
    ):
        """
        Determine whether this track has crossed a counting line this frame.

        Compares the bbox centroid's side (above/below each line) against the
        side recorded in the previous frame.  A side change signals a crossing.

        Returns
        -------
        (crossed, counted) : (bool, bool)
            crossed=True  — the vehicle moved through a line.
            counted=True  — it was also added to the running tally.
        """
        x1, y1, x2, y2 = bbox
        # Use the vertical midpoint of the bbox as the vehicle's representative y.
        cy = (y1 + y2) // 2

        crossed = False
        counted = False

        if self.line_mode == "single":
            # Classify which side of the line the centroid is on this frame.
            side = "above" if cy < self.line_y_a else "below"
            prev = self.track_last_side.get(track_id)
            self.track_last_side[track_id] = side
            # A side change between consecutive frames means the line was crossed.
            if prev is not None and prev != side:
                crossed = True
                # crossed_ids ensures a track ID is counted at most once even if
                # the vehicle oscillates around the line across multiple frames.
                if track_id not in self.crossed_ids:
                    counted = self._handle_new_crossing(
                        track_id, cls_name, bbox, frame
                    )

        else:  # dual
            if track_id not in self.track_last_side:
                self.track_last_side[track_id] = {
                    "a": "above" if cy < self.line_y_a else "below",
                    "b": "above" if cy < self.line_y_b else "below",
                }
                return False, False

            side_a = "above" if cy < self.line_y_a else "below"
            side_b = "above" if cy < self.line_y_b else "below"
            prev_a = self.track_last_side[track_id]["a"]
            prev_b = self.track_last_side[track_id]["b"]

            self.track_last_side[track_id]["a"] = side_a
            self.track_last_side[track_id]["b"] = side_b

            if prev_a != side_a or prev_b != side_b:
                crossed = True
                if track_id not in self.crossed_ids:
                    counted = self._handle_new_crossing(
                        track_id, cls_name, bbox, frame
                    )

        return crossed, counted

    # ------------------------------------------------------------------
    # Counting + ReID
    # ------------------------------------------------------------------

    def _handle_new_crossing(
        self, track_id: int, cls_name: str, bbox: list, frame
    ) -> bool:
        """
        Attempt to count a newly crossing vehicle, applying ReID first.

        Computes a histogram of the vehicle crop and runs _reid_check().  If a
        histogram match is found the crossing is suppressed (returns False) to
        avoid double-counting a re-entering vehicle with a new track ID.
        Otherwise the vehicle is added to all counters and returns True.

        Parameters
        ----------
        track_id : ByteTrack ID for this vehicle.
        cls_name : vehicle class label ("car", "truck", etc.).
        bbox     : [x1, y1, x2, y2] in original frame pixels.
        frame    : full BGR frame used to crop the histogram region.

        Returns
        -------
        bool — True if the vehicle was counted; False if suppressed by ReID.
        """
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            histogram = None
        else:
            histogram = self._compute_histogram(crop)

        if self._reid_check(histogram, bbox):
            logger.info(f"ReID match: track {track_id} skipped")
            return False

        self.crossed_ids.add(track_id)
        self.count_by_class[cls_name] = self.count_by_class.get(cls_name, 0) + 1
        self.flash_registry[track_id] = FLASH_FRAMES

        entry = {
            "id":        track_id,
            "cls":       cls_name,
            "histogram": histogram,
            "bbox":      bbox,
        }

        self.recently_counted.append(entry)
        # Cap the ReID memory at 50 entries to bound memory usage.  Oldest
        # entry is evicted first (FIFO) so vehicles from long ago age out.
        if len(self.recently_counted) > 50:
            self.recently_counted.pop(0)

        logger.info(
            f"Counted: {cls_name} ID={track_id}, "
            f"total={sum(self.count_by_class.values())}"
        )
        return True

    def _compute_histogram(self, crop) -> np.ndarray:
        """
        Compute a normalised 2-D HSV histogram (Hue × Saturation) of a crop.

        Returns a flattened 1-D float32 array suitable for cv2.compareHist().
        """
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Use Hue (channel 0) for colour identity and Saturation (channel 1)
        # for colour purity.  Value (channel 2) is excluded because it varies
        # with lighting conditions and would reduce cross-frame match accuracy.
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [50, 60], [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()

    def _reid_check(self, histogram, bbox: list) -> bool:
        """
        Check whether a histogram matches any recently counted vehicle.

        Iterates recently_counted and computes Pearson correlation
        (HISTCMP_CORREL) between the new histogram and each stored one.
        Returns True (suppress count) if any score exceeds REID_THRESHOLD.

        Parameters
        ----------
        histogram : flattened HSV histogram of the crossing vehicle's crop.
        bbox      : [x1, y1, x2, y2] — included for API symmetry; unused here.

        Returns
        -------
        bool — True if a duplicate match is found; False otherwise.
        """
        if not self.recently_counted:
            return False
        if histogram is None:
            return False

        for entry in self.recently_counted:
            if entry.get("histogram") is None:
                continue
            score = cv2.compareHist(
                histogram,
                entry["histogram"],
                cv2.HISTCMP_CORREL,
            )
            if score > REID_THRESHOLD:
                return True
        return False

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_frame(self, frame, detections: list):
        """
        Annotate a frame with bboxes, labels, counting lines, and flash effects.

        The flash fill is applied before the outline rectangle so the outline
        is always rendered on top and stays crisp regardless of the fill.
        Caller must pass frame.copy() — this method modifies the array in-place.

        Parameters
        ----------
        frame      : np.ndarray — BGR frame to annotate (should be a copy).
        detections : list       — detection dicts from the current frame loop.

        Returns
        -------
        np.ndarray — the annotated frame.
        """
        color_map = {
            "car":        (0, 255, 0),
            "truck":      (0, 0, 255),
            "bus":        (255, 0, 0),
            "motorcycle": (0, 255, 255),
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            track_id = det["track_id"]
            cls_name = det["cls"]
            color    = color_map.get(cls_name, (255, 255, 255))

            if self.flash_registry.get(track_id, 0) > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                # 35% opaque colour fill blended with 65% of the original frame
                # produces a semi-transparent highlight that tints the vehicle
                # without obscuring it.
                cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                self.flash_registry[track_id] -= 1
                if self.flash_registry[track_id] <= 0:
                    del self.flash_registry[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{cls_name} #{track_id}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        total = sum(self.count_by_class.values())

        if self.line_mode == "single":
            cv2.line(
                frame,
                (0, self.line_y_a),
                (self.frame_width, self.line_y_a),
                (255, 255, 255), 2,
            )
            cv2.putText(
                frame, f"Counted: {total}",
                (10, self.line_y_a - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
        else:
            cv2.line(
                frame,
                (0, self.line_y_a),
                (self.frame_width, self.line_y_a),
                (255, 255, 255), 2,
            )
            cv2.putText(
                frame, "Line A",
                (10, self.line_y_a - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.line(
                frame,
                (0, self.line_y_b),
                (self.frame_width, self.line_y_b),
                (255, 255, 255), 2,
            )
            cv2.putText(
                frame, "Line B",
                (10, self.line_y_b - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.putText(
                frame, f"Total Counted: {total}",
                (10, self.frame_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

        return frame

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict:
        """
        Return a summary dict of all counting results accumulated this session.

        Returns
        -------
        dict with keys:
          total_unique   — number of distinct vehicles counted across all lines.
          count_by_class — dict mapping each vehicle class to its count.
          vehicle_log    — list of per-frame detection records (one per box).
        """
        return {
            "total_unique":   len(self.crossed_ids),
            "count_by_class": self.count_by_class,
            "vehicle_log":    self.vehicle_log,
        }
