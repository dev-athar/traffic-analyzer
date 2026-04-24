"""
reid.py — Deep appearance embedding via DeepSORT's MobileNet embedder.

Used in "deepsort" reid_mode to compute per-vehicle appearance vectors and
compare them with cosine similarity, suppressing duplicate counts caused by
occlusion or ID switches more reliably than histogram correlation alone.
"""

import logging
import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)

DEEPSORT_THRESHOLD = 0.90  # cosine similarity above this → same vehicle


class DeepSORTReID:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )
        logger.info("DeepSORT embedder initialized (mobilenet)")

    def get_embedding(self, frame, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        x1c = max(0, x1);  y1c = max(0, y1)
        x2c = min(frame.shape[1], x2);  y2c = min(frame.shape[0], y2)
        crop = frame[y1c:y2c, x1c:x2c]

        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return None

        resized = cv2.resize(crop, (64, 128))
        embedding = self.tracker.embedder.predict([resized])[0]
        return np.array(embedding)

    def compare(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        norm_a = emb_a / (np.linalg.norm(emb_a) + 1e-6)
        norm_b = emb_b / (np.linalg.norm(emb_b) + 1e-6)
        return float(np.dot(norm_a, norm_b))
