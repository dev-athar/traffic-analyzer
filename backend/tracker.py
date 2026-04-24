"""
tracker.py — ByteTrack-backed vehicle tracking, line-crossing counting, and
ReID to suppress duplicate counts after occlusion.
Supports single-line and dual-line counting modes with a flash fill effect
on the bbox when a vehicle crosses a counting line.
"""

import cv2
import numpy as np
import logging
from ultralytics import YOLO
from reid import DeepSORTReID

logger = logging.getLogger(__name__)

# --- Tunable constants ---
LINE_MODE        = "single"
LINE_SINGLE_POS  = 0.50
LINE_DUAL_A_POS  = 0.25
LINE_DUAL_B_POS  = 0.75
FLASH_FRAMES     = 10
MAX_TRACK_AGE    = 50
REID_THRESHOLD   = 0.90
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
        self.reid_mode       = reid_mode
        self.line_mode       = line_mode
        self.line_single_pos = line_single_pos
        self.line_dual_a_pos = line_dual_a_pos
        self.line_dual_b_pos = line_dual_b_pos

        self.model = YOLO("yolov8n.pt")

        self.crossed_ids      = set()
        self.track_last_side  = {}
        self.recently_counted = []
        self.vehicle_log      = []
        self.count_by_class   = {}
        self.flash_registry   = {}

        self.line_y_a = None
        self.line_y_b = None

        if self.reid_mode == "accurate":
            self.deepsort_reid = DeepSORTReID()
            logger.info("DeepSORT ReID loaded")
        else:
            self.deepsort_reid = None

        logger.info(
            f"VehicleTracker initialized, "
            f"reid_mode={reid_mode}, "
            f"line_mode={line_mode}"
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_frame_dimensions(self, width: int, height: int):
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
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) // 2

        crossed = False
        counted = False

        if self.line_mode == "single":
            side = "above" if cy < self.line_y_a else "below"
            prev = self.track_last_side.get(track_id)
            self.track_last_side[track_id] = side
            if prev is not None and prev != side:
                crossed = True
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
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            histogram = None
        else:
            histogram = self._compute_histogram(crop)

        if self._reid_check(histogram, bbox, frame):
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
            "embedding": None,
        }

        if self.reid_mode == "accurate" and self.deepsort_reid is not None:
            entry["embedding"] = self.deepsort_reid.get_embedding(frame, bbox)

        self.recently_counted.append(entry)
        if len(self.recently_counted) > 50:
            self.recently_counted.pop(0)

        logger.info(
            f"Counted: {cls_name} ID={track_id}, "
            f"total={sum(self.count_by_class.values())}"
        )
        return True

    def _compute_histogram(self, crop) -> np.ndarray:
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [50, 60], [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()

    def _reid_check(self, histogram, bbox: list, frame) -> bool:
        if not self.recently_counted:
            return False
        if histogram is None:
            return False

        if self.reid_mode == "accurate" and self.deepsort_reid is not None:
            new_emb = self.deepsort_reid.get_embedding(frame, bbox)
            if new_emb is not None:
                for entry in self.recently_counted:
                    if entry.get("embedding") is None:
                        continue
                    score = self.deepsort_reid.compare(new_emb, entry["embedding"])
                    if score > 0.70:
                        logger.info(
                            f"DeepSORT ReID match: score={score:.3f}, skipping"
                        )
                        return True
                return False
            else:
                logger.warning(
                    "DeepSORT embedding failed, falling back to histogram"
                )

        # Fast mode OR DeepSORT fallback
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
        return {
            "total_unique":   len(self.crossed_ids),
            "count_by_class": self.count_by_class,
            "vehicle_log":    self.vehicle_log,
        }
