# Smart Drone Traffic Analyzer

Detects, tracks, and counts vehicles from static drone footage. Upload a video, draw your counting lines, get results with bounding boxes, per-class counts, and exportable reports.

**Stack:** React + Vite / FastAPI / YOLOv8 / ByteTrack

---

## Setup

**Backend**
```bash
cd backend
python -m venv venv && source venv\Scripts\activate   # MacOS: venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install && npm run dev
```

Open `http://localhost:5173`. YOLOv8 weights download automatically on first run.

---

## How the System Works

The user uploads an `.mp4` through the browser and configures two things: a detection mode (Fast or Accurate) and the position of the counting lines. On submit, the frontend POSTs the file as multipart form data to the backend, which saves it, generates a job ID, and immediately starts processing in a background thread.

While processing runs, the frontend holds a WebSocket connection open. The backend pushes `{ status, progress }` every half second — no polling. The progress bar updates live. When the job finishes, the frontend navigates to the results page automatically.

The results page fetches the stats JSON and points an HTML5 video player at the processed file, which FastAPI serves as a static file from the `outputs/` directory. Reports are available as direct file downloads via query parameter (`?format=csv` or `?format=excel`).

Everything is stateless between requests. Job data lives in an in-memory Python dict — simple, no database, adequate for local single-session use.

---

## The Approach

### Why static drone footage

Moving drone footage requires frame-by-frame camera motion compensation — homography estimation or optical flow — just to stabilize the image before any detection can happen. That is computationally expensive and introduces its own failure modes. Static footage eliminates this entire layer. The drone hovers, the camera does not move, and the road geometry stays fixed in pixel space for the full duration of the video.

That one constraint is what makes the rest of the system simple and efficient.

### The counting method

Instead of tracking full vehicle trajectories, detecting travel direction, or defining entry/exit zones — all of which add complexity and break under occlusion — we take a simpler approach: draw horizontal lines across the frame and count any vehicle that crosses one.

```
════════════  LINE A  (default 25% from top)  ════════════

                [ traffic flows through here ]

════════════  LINE B  (default 75% from top)  ════════════
```

Two lines, not one. A vehicle is counted the moment its bounding box centroid crosses either line. Its ID goes into a `crossed_ids` set. If the same ID crosses the other line later, it is ignored — already counted.

The reason for two lines: a single line at 50% silently misses vehicles that are already in the frame when recording starts (they sit below the line and never cross it) and vehicles that enter near the end (they do not reach the line before recording stops). Dual lines catch both — vehicles present at start will cross Line B, vehicles entering late will cross Line A. Both positions are adjustable via sliders in the UI.

This works for one-way roads, bidirectional roads, and multi-lane traffic without any configuration beyond setting the line height.

---

## How Detection and Tracking Work

### Detection

YOLOv8 pretrained on COCO runs on every frame. It detects four classes: car, truck, bus, motorcycle. Two modes are available:

| Mode | Model | Resolution | Conf. | What changes |
|------|-------|------------|-------|--------------|
| Fast | YOLOv8n (nano) | 640px | 0.25 | Faster. Good for low-altitude footage with large vehicles |
| Accurate | YOLOv8s (small) | 1280px | 0.35 | Slower. Catches small and distant vehicles that nano misses |

We considered YOLOv8m and YOLOv8l. Both improve accuracy further but are 3–10x slower on CPU with diminishing returns for this use case. YOLOv8s at 1280px is the practical ceiling before processing time becomes unreasonable.

### Tracking

ByteTrack assigns a persistent ID to each vehicle and maintains it across frames using Kalman-predicted motion. It is built into Ultralytics — no extra dependency. We evaluated DeepSORT, SORT, and StrongSORT. ByteTrack won because it handles occlusion well, runs fast, and does not need an appearance model for frame-to-frame tracking.

`max_age` is set to 50 frames (default is 30). This gives ByteTrack more time to re-associate a vehicle that disappears behind a bridge or a larger truck. A track that survives occlusion keeps its original ID, so the counting logic never sees a problem. The Re-ID check below only activates when ByteTrack gives up and assigns a new ID.

### Re-ID: preventing double counts after long occlusion

When ByteTrack loses a vehicle for more than 50 frames and it reappears, it gets a new ID. The system now sees what looks like a new vehicle crossing a line. If it counts it, that is a double count.

Every crossing event triggers a Re-ID check before the count is accepted. The system crops the bounding box, computes an HSV color histogram (H and S channels, 50×60 bins), and compares it against a rolling list of the last 50 counted vehicles using `cv2.compareHist` with correlation mode. If similarity exceeds 0.90, the crossing is rejected as a duplicate.

HSV over RGB because it separates color from brightness — robust to the lighting and shadow shifts in outdoor drone footage. The 0.90 threshold was calibrated by testing: lower values false-matched different vehicles of the same color, higher values missed actual duplicates.

The accepted tradeoff: two genuinely different vehicles with near-identical appearance (same make, model, color) may cause one to be skipped. Resolving this requires license plate OCR, which is out of scope.

---

## Visual Output

The processed video includes bounding boxes color-coded by class (car=green, truck=red, bus=blue, motorcycle=yellow) with track IDs above each box. When a vehicle is counted, its box fills with a semi-transparent class-colored overlay for 10 frames — a visible flash confirming the count. Both counting lines are drawn in white with a live total overlay.

---

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/upload` | `.mp4` + config → `job_id` |
| GET | `/status/{job_id}` | Progress % |
| GET | `/result/{job_id}` | Stats JSON |
| GET | `/video/{job_id}` | Processed `.mp4` stream |
| GET | `/report/{job_id}?format=csv\|excel` | Report download |
| WS | `/ws/{job_id}` | Live progress push |

---

## Reports

Generated automatically on completion:

- **CSV** — One row per detection: frame index, timestamp, track ID, class, confidence, bbox coordinates, crossed line (bool), counted (bool)
- **Excel** — Same detections sheet + Summary sheet: total count, count by class, processing duration, mode used, video FPS, resolution, frame count

---

## Project Structure

```
smart-drone-analyzer/
├── backend/
│   ├── main.py          # FastAPI, routes, WebSocket, job store
│   ├── processor.py     # Frame loop, video I/O, progress
│   ├── tracker.py       # YOLOv8 + ByteTrack + line crossing + ReID
│   ├── reid.py          # DeepSORT embedder (accurate mode)
│   ├── reporter.py      # CSV/Excel generation
│   ├── models.py        # Pydantic schemas
│   └── requirements.txt
├── frontend/src/
│   ├── components/
│   │   ├── Upload.jsx   # File drop, config, line preview
│   │   ├── Progress.jsx # WebSocket progress bar
│   │   └── Results.jsx  # Video player, stats, downloads
│   ├── App.jsx
│   └── api.js
├── uploads/             # Raw videos (git-ignored)
├── outputs/             # Processed output (git-ignored)
└── README.md
```
