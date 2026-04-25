# Traffic Analyzer

## 1. Project Summary

Traffic Analyzer accepts an MP4 video recorded from a static drone and counts how many vehicles pass through one or two user-defined horizontal counting lines. The output is an annotated MP4 (bounding boxes, line overlays, running count), a CSV, and an Excel workbook. Internally, YOLOv8 detects vehicles each frame, ByteTrack assigns persistent track IDs across frames, centroid-based line-crossing logic fires the counter, and an HSV-histogram ReID check suppresses double counts when a vehicle re-enters the scene with a new track ID.

---

## 2. Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React 18 + Vite 5 | Single-page UI: upload, progress, results |
| HTTP client | Axios 1.7 | REST calls from browser to backend |
| Routing | react-router-dom 6 | Client-side navigation between Upload / Progress / Results |
| Backend | FastAPI + uvicorn | REST API, WebSocket server, static file serving |
| Detection | Ultralytics YOLOv8n / YOLOv8s | Vehicle detection (COCO classes 2, 3, 5, 7) |
| Tracking | ByteTrack (via Ultralytics) | Persistent track IDs across frames |
| Video I/O | OpenCV (cv2) | Frame decode, annotation, H.264 encode |
| Reporting | Python `csv` stdlib + openpyxl | CSV and .xlsx report generation |
| Communication | HTTP REST + WebSocket | Upload/result via HTTP; live progress via WS |

---

## 3. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 or higher | Required by Ultralytics |
| Node.js | 18 or higher | Required by Vite 5 |
| npm | 9 or higher | Bundled with Node 18 |
| OpenCV H.264 support | — | `avc1` codec; included in the `opencv-python` wheel on most platforms |

**YOLOv8 weights** (`yolov8n.pt` and `yolov8s.pt`) are downloaded automatically by Ultralytics on first use and cached in the working directory. No manual download is needed.

---

## 4. Installation

### 4.1 Clone / obtain the repository

```
git clone <repo-url>
cd smart-drone-analyzer
```

### 4.2 Backend — Python virtual environment

**Windows**
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4.3 Frontend — Node dependencies

Open a second terminal (keep the backend venv terminal open):
```
cd frontend
npm install
```

### 4.4 Required directories

The backend creates `uploads/` and `outputs/` automatically on startup via `Path.mkdir(exist_ok=True)`, but you can create them manually to confirm they will land in the right place:

```
# Run from inside the backend/ directory
mkdir -p uploads outputs
```

Both directories must exist relative to the working directory from which `uvicorn` is started (i.e. `backend/`). All file I/O in `main.py`, `processor.py`, and `reporter.py` uses relative paths (`./uploads`, `./outputs`, `outputs/`).

---

## 5. Running the Application

Both servers must run simultaneously in separate terminals.

**Terminal 1 — Backend (port 8000)**
```
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend (port 5173)**
```
cd frontend
npm run dev
```

Open `http://localhost:5173` in a browser.

**Port assignments**

| Server | Port | URL |
|---|---|---|
| FastAPI / uvicorn | 8000 | `http://localhost:8000` |
| Vite dev server | 5173 | `http://localhost:5173` |

**Proxy note:** `vite.config.js` defines a proxy for the `/api` prefix only — requests from the browser to `/api/*` are rewritten (the `/api` prefix is stripped) and forwarded to `http://localhost:8000`. This proxy is used exclusively by the report download buttons in `Results.jsx` (`/api/report/<jobId>?format=csv`). All other frontend traffic (axios calls, WebSocket, video `src`) goes directly to `http://localhost:8000` via hardcoded `baseURL` in `api.js`.

---

## 6. Full Application Flow

### Step 1 — User selects file and config in `Upload.jsx`

The `Upload` component holds the following state:

| State variable | Type | Default | Holds |
|---|---|---|---|
| `file` | `File \| null` | `null` | Selected MP4 `File` object |
| `reidMode` | `string` | `"fast"` | `"fast"` or `"accurate"` |
| `lineMode` | `string` | `"dual"` | `"single"` or `"dual"` |
| `lineSinglePos` | `number` | `50` | Integer 10–90 representing % from top |
| `lineDualA` | `number` | `45` | Integer 10–90 for Line A position |
| `lineDualB` | `number` | `80` | Integer 10–90 for Line B position |
| `previewFrame` | `string \| null` | `null` | data URL of a JPEG frame extracted at 10% into the video |

When a file is selected (via `onSelectFile`), the component creates a temporary `<video>` element, seeks to 10% of the video duration, and draws that frame onto a `<canvas>` to produce `previewFrame`. The `LinePreview` sub-component then re-renders the canvas each time a slider moves, drawing the counting line(s) on top of the preview frame so the user can see exactly where lines will fall.

`updateDualA` and `updateDualB` enforce the constraint that Line A must stay above Line B by clamping values before calling `setLineDualA` / `setLineDualB`.

When the user clicks **INITIALIZE ANALYSIS**, `handleUpload` is called:
1. Guards against missing file.
2. Sets `uploading = true` (triggers the animated dots in the button label).
3. Calls `uploadVideo()` from `api.js`, dividing slider integers by 100 to convert to fractional values (`lineSinglePos / 100`, `lineDualA / 100`, `lineDualB / 100`).
4. On success, calls `navigate('/processing/<job_id>')` with `state` containing the config values.
5. On error, writes `err.response?.data?.detail` to the `error` state.

The `uploadVideo` function in `api.js` builds a `FormData` object containing:

```
file            → the MP4 File object
reid_mode       → "fast" | "accurate"
line_mode       → "single" | "dual"
line_single_pos → float 0.0–1.0
line_dual_a_pos → float 0.0–1.0
line_dual_b_pos → float 0.0–1.0
```

This is POSTed to `http://localhost:8000/upload` with `Content-Type: multipart/form-data`.

---

### Step 2 — `POST /upload` hits the backend

**Handler:** `upload_video()` in `main.py`

1. Validates that the filename ends with `.mp4`; raises HTTP 400 otherwise.
2. Generates `job_id = str(uuid.uuid4())[:8]` — first 8 hex characters of a UUID4.
3. Saves the file to `uploads/<job_id>_<original_filename>` (e.g. `uploads/3e025c2d_traffic.mp4`).
4. Creates the `jobs[job_id]` entry:

```python
jobs[job_id] = {
    "status":          "pending",       # str
    "progress":        0.0,             # float 0.0–100.0
    "error":           None,            # str | None
    "summary":         None,            # dict | None (filled on completion)
    "csv_path":        None,            # str | None
    "excel_path":      None,            # str | None
    "reid_mode":       reid_mode,       # "fast" | "accurate"
    "line_mode":       line_mode,       # "single" | "dual"
    "line_single_pos": line_single_pos, # float
    "line_dual_a_pos": line_dual_a_pos, # float
    "line_dual_b_pos": line_dual_b_pos, # float
    "created_at":      time.time(),     # float (Unix timestamp)
    "filename":        file.filename,   # str
    "video_path":      str(dest),       # str (absolute path to saved file)
}
```

5. Calls `background_tasks.add_task(run_processing, job_id)` — queues `run_processing` to run in a background thread managed by FastAPI.
6. Returns an `UploadResponse` JSON:

```json
{
  "job_id":          "3e025c2d",
  "filename":        "traffic.mp4",
  "reid_mode":       "fast",
  "line_mode":       "dual",
  "line_single_pos": 0.5,
  "line_dual_a_pos": 0.45,
  "line_dual_b_pos": 0.8,
  "status":          "pending"
}
```

---

### Step 3 — Frontend receives `job_id`

`handleUpload` in `Upload.jsx` receives `data.job_id` and immediately calls:
```
navigate(`/processing/${data.job_id}`, { state: { reidMode, lineMode, ... } })
```
React Router renders the `Progress` component at the route `/processing/:jobId`.

---

### Step 4 — WebSocket connection opens

`Progress.jsx` mounts and its `useEffect` hook calls `hydrateStatus()` then `connect()`.

**`hydrateStatus()`** calls `getStatus(jobId)` from `api.js` (a GET to `/status/<jobId>`) to immediately populate status and progress from the HTTP response, in case the WebSocket takes a moment to connect.

**`connect()`** calls `connectProgressWS(jobId, onMessage, onError)` from `api.js`, which opens:
```
ws://localhost:8000/ws/<jobId>
```

**Backend handler:** `websocket_progress()` in `main.py`

The backend enters an `asyncio` loop:
1. Looks up `jobs[job_id]`.
2. Sends a JSON message over the socket:

```json
{
  "status":   "processing",
  "progress": 47.3,
  "error":    null
}
```

3. If `status` is `"complete"` or `"error"`, breaks out of the loop and the socket closes naturally.
4. Otherwise waits `asyncio.sleep(0.5)` (500 ms) before the next iteration.

**Frontend handling per message (`onMessage` in `Progress.jsx`):**

- If `data.progress` is a number: calls `setProgress(Math.round(data.progress))`.
- If `data.status` is present: calls `setStatus(data.status)`.
- If `data.error` is present: calls `setError(data.error)`.
- If `data.status === "complete"`: schedules a `setTimeout` for 1500 ms, then calls `navigate('/result/<jobId>')`.

The progress bar renders `<div style={{ width: '<progress>%' }} />` directly from the `progress` state.

---

### Step 5 — Background thread: CV pipeline starts

**`run_processing(job_id)`** in `main.py` runs in a background thread:

1. Sets `job["status"] = "processing"`.
2. Defines `progress_callback(pct)` as a closure that writes `pct` into `job["progress"]`.
3. Calls `process_video()` in `processor.py` with the stored job config.

**`process_video()`** in `processor.py`, step by step:

1. `cap = cv2.VideoCapture(video_path)` — opens the source video.
2. Reads metadata: `total_frames`, `fps`, `width`, `height` via `cap.get(...)`.
3. Opens an H.264 `VideoWriter`:
   ```python
   fourcc = cv2.VideoWriter_fourcc(*"avc1")
   out = cv2.VideoWriter(f"outputs/{job_id}_processed.mp4", fourcc, fps, (width, height))
   ```
4. Instantiates `VehicleTracker(reid_mode, line_mode, line_single_pos, line_dual_a_pos, line_dual_b_pos)`.
5. Calls `tracker.set_frame_dimensions(width, height)` to convert fractional line positions to pixel y-coordinates.
6. Enters the frame loop: `cap.read()` returns `(ret, frame)` each iteration.
7. For every frame where `frame_index % FRAME_SKIP == 0` (currently `FRAME_SKIP = 1`, so every frame):
   - Calls `annotated, _ = tracker.process_frame(frame, frame_index, fps)`.
   - Writes `annotated` to `out`.
   - Skipped frames (when `FRAME_SKIP > 1`) write the raw `frame` so output timing matches source.
8. Every 10 frames: calls `progress_callback(round((frame_index / total_frames) * 100, 1))`.
9. After the loop: calls `cap.release()` and `out.release()`.
10. Calls `tracker.get_summary()` to get counting results.
11. Augments summary with job metadata (`job_id`, `fps`, `width`, `height`, `total_frames`, `processed_frames`, file paths, config values).
12. Returns the summary dict.

Back in `run_processing()`:
- `job["csv_path"]  = generate_csv(vehicle_log, result, job_id)`
- `job["excel_path"] = generate_excel(vehicle_log, result, job_id)`
- `job["summary"] = result`
- `job["status"] = "complete"`
- `job["progress"] = 100.0`

---

### Step 6 — Inside `process_frame()` — per frame

**`tracker.process_frame(frame, frame_index, fps)`** in `tracker.py`:

1. **In `"accurate"` mode only:** downsizes the frame to at most 1280px wide before inference, stores `scale_x` and `scale_y` ratios for coordinate mapping back.

2. **`model.track()`** is called with `persist=True`, `classes=[2, 3, 5, 7]`, `tracker="bytetrack.yaml"`, `verbose=False`. In accurate mode, also `imgsz=1280, conf=0.35`.
   - `persist=True` tells ByteTrack to carry track state forward from the previous call so IDs are stable frame-to-frame.
   - `classes=[2, 3, 5, 7]` filters to car, motorcycle, bus, truck (COCO class IDs).

3. `results[0].boxes` contains all detected boxes. Each `box` exposes `.id` (track ID), `.cls` (class index), `.conf` (confidence), `.xyxy[0]` (bounding box in pixel coordinates).

4. For each `box`:
   - Skip if `box.id is None` (ByteTrack hasn't committed an ID yet for that track).
   - Extract `track_id`, `cls_id`, `confidence`, `x1, y1, x2, y2`.
   - In accurate mode: scale coordinates back to original frame space (`x1 *= scale_x`, etc.).
   - Map `cls_id` to `cls_name` via `cls_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}`.

5. **`_check_line_crossing(track_id, cls_name, confidence, [x1,y1,x2,y2], frame)`:**
   - Computes centroid `cy = (y1 + y2) // 2`.
   - In `"single"` mode: determines `side = "above"` if `cy < self.line_y_a` else `"below"`. Compares to `self.track_last_side.get(track_id)` (previous frame's side). Updates `track_last_side[track_id]`. A side change means `crossed = True`. If `track_id not in self.crossed_ids`, calls `_handle_new_crossing()`.
   - In `"dual"` mode: calculates side relative to both `line_y_a` and `line_y_b` independently. On first appearance, records both sides and returns `(False, False)`. On subsequent frames, a side change on either line triggers `crossed = True`.
   - Returns `(crossed: bool, counted: bool)`.

6. **`_handle_new_crossing(track_id, cls_name, bbox, frame)`:**
   - Crops `frame[y1:y2, x1:x2]`.
   - If crop is non-empty: calls `_compute_histogram(crop)` to get a normalised HSV histogram.
   - Calls `_reid_check(histogram, bbox)`:
     - Iterates `self.recently_counted`. For each entry, calls `cv2.compareHist(..., cv2.HISTCMP_CORREL)`.
     - If any score exceeds `REID_THRESHOLD (0.90)`: returns `True` → suppress count.
     - Returns `False` if no match found.
   - If ReID returns `True`: logs the suppression, returns `False` (not counted).
   - If ReID returns `False`: adds `track_id` to `self.crossed_ids`. Increments `self.count_by_class[cls_name]`. Sets `self.flash_registry[track_id] = FLASH_FRAMES (10)`. Appends `{"id", "cls", "histogram", "bbox"}` to `self.recently_counted` (capped at 50 entries FIFO). Returns `True`.

7. **`_compute_histogram(crop)`:**
   - Converts crop from BGR to HSV: `cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)`.
   - Computes a 2D histogram over channel 0 (Hue, 50 bins, 0–180) and channel 1 (Saturation, 60 bins, 0–256). Value channel excluded (lighting-dependent).
   - Normalises with `cv2.NORM_MINMAX` to range 0–1.
   - Returns the flattened 1D `float32` array.

8. A `detections` dict is appended for each box and a `vehicle_log` entry is also appended:

```python
vehicle_log entry = {
    "frame_index":   int,
    "timestamp_sec": float,
    "track_id":      int,
    "vehicle_class": str,
    "confidence":    float,
    "bbox_x1": int, "bbox_y1": int,
    "bbox_x2": int, "bbox_y2": int,
    "crossed_line":  bool,
    "counted":       bool,
}
```

9. **`_draw_frame(frame.copy(), detections)`** is called last:
   - For each detection: if `flash_registry.get(track_id, 0) > 0`, blends a solid colour fill (35% opacity) over the bbox using `cv2.addWeighted`, then decrements the counter.
   - Draws a 2px outline rectangle in the class colour: green (car), red (truck), blue (bus), yellow (motorcycle).
   - Draws label text `"<cls_name> #<track_id>"` above the box.
   - Draws counting line(s) as white horizontal lines across the full frame width.
   - In single mode: prints `"Counted: <total>"` above the line.
   - In dual mode: prints `"Line A"` / `"Line B"` labels and `"Total Counted: <total>"` at the bottom of the frame.
   - Returns the annotated frame.

10. `process_frame` returns `(annotated, detections)`. The caller (`process_video`) uses only `annotated` and discards `detections`.

---

### Step 7 — Processing completes

After `process_video()` returns:

1. `run_processing()` calls `generate_csv(vehicle_log, result, job_id)`:
   - Calls `_build_counted_rows(vehicle_log)` which filters to `counted=True` rows, deduplicates by `track_id` (FIFO), and assigns a sequential `counter` integer.
   - Writes `outputs/<job_id>_report.csv` with a header block and data rows.
   - Returns the file path string.

2. Calls `generate_excel(vehicle_log, result, job_id)`:
   - Same `_build_counted_rows()` call.
   - Sheet `"Counted"`: same columns as CSV.
   - Sheet `"Summary"`: aggregate metrics (total vehicles, per-class counts, duration, reid_mode, line_mode, fps, resolution, frame counts).
   - Calls `_autosize_worksheet_columns()` on both sheets.
   - Saves as `outputs/<job_id>_report.xlsx`.

3. Sets:
   ```python
   job["csv_path"]   = <path string>
   job["excel_path"] = <path string>
   job["summary"]    = result
   job["status"]     = "complete"
   job["progress"]   = 100.0
   job["error"]      = None
   ```

4. On the next WebSocket loop iteration (within 500 ms), `websocket_progress()` reads `job["status"] == "complete"`, sends the final `{"status": "complete", "progress": 100.0, "error": null}` message, then breaks out of the loop.

---

### Step 8 — Frontend receives `status: complete`

`Progress.jsx` receives the final WebSocket message. The `onMessage` handler detects `data.status === "complete"` and calls:
```javascript
window.setTimeout(() => navigate(`/result/${jobId}`), 1500)
```
After a 1.5-second pause (to let the user see "ANALYSIS COMPLETE"), it navigates to the results page.

---

### Step 9 — Results page loads

`Results.jsx` mounts and its `useEffect` calls `fetchData()`, which calls `getResult(jobId)` from `api.js` — a GET to `http://localhost:8000/result/<jobId>`.

The backend `get_result()` handler returns a `ResultResponse`:

```json
{
  "job_id":                   "3e025c2d",
  "status":                   "complete",
  "total_unique":             42,
  "count_by_class":           {"car": 30, "truck": 7, "bus": 3, "motorcycle": 2},
  "processing_duration_sec":  18.4,
  "reid_mode":                "fast",
  "line_mode":                "dual",
  "fps":                      25.0,
  "width":                    1920,
  "height":                   1080,
  "total_frames":             500,
  "processed_frames":         500,
  "output_video_url":         "/static/3e025c2d_processed.mp4",
  "report_csv_url":           "/static/3e025c2d_report.csv",
  "report_excel_url":         "/static/3e025c2d_report.xlsx"
}
```

The annotated video is displayed in a `<video controls>` element with `src` hardcoded to:
```
http://localhost:8000/static/<jobId>_processed.mp4
```
The `/static` mount in FastAPI serves the `outputs/` directory with `Accept-Ranges: bytes` and `Cache-Control: no-cache` headers (added by the `add_static_headers` middleware), enabling byte-range requests needed for browser video seeking.

Download buttons use `window.open(...)` with the Vite-proxied path:
```
/api/report/<jobId>?format=csv
/api/report/<jobId>?format=excel
```
Vite strips `/api` and forwards to `http://localhost:8000/report/<jobId>?format=csv`, which returns a `FileResponse` with the appropriate MIME type and `filename` header triggering a browser download.

---

## 7. Component & Function Reference

### Backend Functions

| File | Function | Purpose | Called By |
|---|---|---|---|
| `main.py` | `upload_video()` | Validates .mp4, saves file, creates `jobs` entry, queues background task | FastAPI route `POST /upload` |
| `main.py` | `run_processing()` | Drives full pipeline in background thread; updates `jobs` dict throughout | `BackgroundTasks.add_task` |
| `main.py` | `websocket_progress()` | Streams `{status, progress, error}` every 500 ms until job finishes | FastAPI route `WS /ws/{job_id}` |
| `main.py` | `get_status()` | Returns current `StatusResponse` for a job | FastAPI route `GET /status/{job_id}` |
| `main.py` | `get_result()` | Returns full `ResultResponse` once status is `"complete"`; returns 425 if not ready | FastAPI route `GET /result/{job_id}` |
| `main.py` | `get_video()` | Returns `FileResponse` for the processed MP4 | FastAPI route `GET /video/{job_id}` |
| `main.py` | `get_report()` | Returns CSV or Excel `FileResponse` based on `?format=` param | FastAPI route `GET /report/{job_id}` |
| `main.py` | `add_static_headers()` | Middleware adding byte-range and no-cache headers to `/static` responses | Every `/static` request |
| `processor.py` | `process_video()` | Opens video, runs frame loop, writes annotated output, returns summary dict | `run_processing()` |
| `tracker.py` | `VehicleTracker.__init__()` | Loads YOLO model, initialises all state stores (`crossed_ids`, `track_last_side`, `recently_counted`, `vehicle_log`, `count_by_class`, `flash_registry`) | `process_video()` |
| `tracker.py` | `set_frame_dimensions()` | Converts fractional line positions to pixel y-coordinates; stores `frame_width`, `frame_height` | `process_video()` |
| `tracker.py` | `process_frame()` | Runs `model.track()`, extracts boxes, calls crossing logic, draws frame | `process_video()` |
| `tracker.py` | `_check_line_crossing()` | Computes centroid side vs. stored previous side; triggers `_handle_new_crossing()` on change | `process_frame()` |
| `tracker.py` | `_handle_new_crossing()` | Computes histogram, runs ReID check, updates all counters and registers | `_check_line_crossing()` |
| `tracker.py` | `_compute_histogram()` | Returns normalised flattened HSV (Hue×Saturation) histogram of a vehicle crop | `_handle_new_crossing()` |
| `tracker.py` | `_reid_check()` | Compares histogram against `recently_counted` via Pearson correlation; returns `True` to suppress | `_handle_new_crossing()` |
| `tracker.py` | `_draw_frame()` | Annotates frame in-place: flash fills, bbox outlines, labels, counting lines, totals | `process_frame()` |
| `tracker.py` | `get_summary()` | Returns `{total_unique, count_by_class, vehicle_log}` | `process_video()` |
| `reporter.py` | `generate_csv()` | Writes `outputs/<job_id>_report.csv`; returns file path | `run_processing()` |
| `reporter.py` | `generate_excel()` | Writes `outputs/<job_id>_report.xlsx` with Counted + Summary sheets; returns file path | `run_processing()` |
| `reporter.py` | `_build_counted_rows()` | Deduplicates `vehicle_log` down to one row per unique counted crossing | `generate_csv()`, `generate_excel()` |
| `reporter.py` | `_autosize_worksheet_columns()` | Sets each column width to match its longest cell value | `generate_excel()` |

### Frontend Functions & Components

| File | Function / Component | Purpose | Connects To |
|---|---|---|---|
| `main.jsx` | `main` (entry point) | Mounts `<App>` into `#root` | `App.jsx` |
| `App.jsx` | `App` | Top-level router; renders nav bar and three routes | React Router |
| `Upload.jsx` | `Upload` | Upload form: file picker, config sliders, preview | `api.js → uploadVideo()` |
| `Upload.jsx` | `onSelectFile()` | Validates .mp4, extracts frame at 10% for preview | Called on file input change / drop |
| `Upload.jsx` | `handleUpload()` | Calls `uploadVideo()`, navigates to Progress on success | `api.js → uploadVideo()` |
| `Upload.jsx` | `updateDualA()` | Updates Line A slider; enforces Line A < Line B | Slider `onChange` |
| `Upload.jsx` | `updateDualB()` | Updates Line B slider; enforces Line B > Line A | Slider `onChange` |
| `Upload.jsx` | `clearFile()` | Clears file, preview, and file input value | "REMOVE" button |
| `Upload.jsx` | `LinePreview` | Sub-component; draws counting line(s) on a `<canvas>` over the preview frame | `useEffect` re-runs on any config change |
| `Progress.jsx` | `Progress` | Shows progress bar, percent, status message | `api.js → connectProgressWS()`, `getStatus()` |
| `Progress.jsx` | `hydrateStatus()` | Polls `GET /status/<jobId>` once on mount to seed initial state | `api.js → getStatus()` |
| `Progress.jsx` | `connect()` | Opens WebSocket; `onMessage` updates state; detects `"complete"` and navigates | `api.js → connectProgressWS()` |
| `Results.jsx` | `Results` | Displays annotated video, count stats, run metadata, download buttons | `api.js → getResult()` |
| `Results.jsx` | `fetchData()` | Calls `getResult(jobId)` and stores result in component state | `api.js → getResult()` |
| `api.js` | `uploadVideo()` | Builds `FormData`, POSTs to `/upload`, returns `UploadResponse` data | `Upload.jsx → handleUpload()` |
| `api.js` | `getStatus()` | GET `/status/<jobId>`; returns `StatusResponse` data | `Progress.jsx → hydrateStatus()` |
| `api.js` | `getResult()` | GET `/result/<jobId>`; returns `ResultResponse` data | `Results.jsx → fetchData()` |
| `api.js` | `connectProgressWS()` | Opens `WebSocket` to `ws://localhost:8000/ws/<jobId>`; wires `onmessage`, `onerror`, `onclose` | `Progress.jsx → connect()` |

---

## 8. Data Structures

### `jobs{}` store entry (in-memory, `main.py`)

```python
{
    "status":          str,        # "pending" | "processing" | "complete" | "error"
    "progress":        float,      # 0.0 to 100.0
    "error":           str | None, # exception message if status == "error"
    "summary":         dict | None,# full result dict from process_video() once complete
    "csv_path":        str | None, # absolute path to outputs/<job_id>_report.csv
    "excel_path":      str | None, # absolute path to outputs/<job_id>_report.xlsx
    "reid_mode":       str,        # "fast" | "accurate"
    "line_mode":       str,        # "single" | "dual"
    "line_single_pos": float,      # 0.0–1.0
    "line_dual_a_pos": float,      # 0.0–1.0
    "line_dual_b_pos": float,      # 0.0–1.0
    "created_at":      float,      # Unix timestamp from time.time()
    "filename":        str,        # original uploaded filename
    "video_path":      str,        # absolute path to uploads/<job_id>_<filename>
}
```

### `vehicle_log` entry (`tracker.py`, accumulated in `VehicleTracker.vehicle_log`)

One entry is appended per detected bounding box per processed frame:

```python
{
    "frame_index":   int,   # zero-based frame number
    "timestamp_sec": float, # frame_index / fps, rounded to 3 decimal places
    "track_id":      int,   # ByteTrack persistent track ID
    "vehicle_class": str,   # "car" | "truck" | "bus" | "motorcycle"
    "confidence":    float, # YOLO detection confidence, rounded to 3 d.p.
    "bbox_x1":       int,   # bounding box top-left x (original frame pixels)
    "bbox_y1":       int,   # bounding box top-left y
    "bbox_x2":       int,   # bounding box bottom-right x
    "bbox_y2":       int,   # bounding box bottom-right y
    "crossed_line":  bool,  # True if centroid changed sides this frame
    "counted":       bool,  # True if this event incremented the running total
}
```

### `recently_counted` entry (`tracker.py`)

Stored in `VehicleTracker.recently_counted` (max 50, FIFO):

```python
{
    "id":        int,              # track_id at time of counting
    "cls":       str,              # vehicle class label
    "histogram": np.ndarray | None,# flattened float32 HSV histogram (3000 elements: 50×60)
    "bbox":      list,             # [x1, y1, x2, y2] at time of crossing
}
```

### WebSocket message shape (backend → frontend)

```json
{
    "status":   "pending | processing | complete | error",
    "progress": 0.0,
    "error":    null
}
```

Note: The backend only sends these three fields. `Progress.jsx` also checks for `data.reid_mode`, `data.line_mode`, `data.processed_frames`, and `data.total_frames` in its `onMessage` handler, but the current backend does not include those fields in WebSocket messages — they would only appear if the backend is extended.

### Pydantic models (`models.py`)

**`UploadResponse`**
```python
{
    "job_id":          str,
    "filename":        str,
    "reid_mode":       str,
    "line_mode":       str,
    "line_single_pos": float,
    "line_dual_a_pos": float,
    "line_dual_b_pos": float,
    "status":          str,
}
```

**`StatusResponse`**
```python
{
    "job_id":   str,
    "status":   str,
    "progress": float,
    "error":    str | None,
}
```

**`ResultResponse`**
```python
{
    "job_id":                   str,
    "status":                   str,
    "total_unique":             int,
    "count_by_class":           dict,   # {"car": int, "truck": int, ...}
    "processing_duration_sec":  float,
    "reid_mode":                str,
    "line_mode":                str,
    "fps":                      float,
    "width":                    int,
    "height":                   int,
    "total_frames":             int,
    "processed_frames":         int,
    "output_video_url":         str,    # "/static/<job_id>_processed.mp4"
    "report_csv_url":           str,    # "/static/<job_id>_report.csv"
    "report_excel_url":         str,    # "/static/<job_id>_report.xlsx"
}
```

---

## 9. API Reference

### HTTP Endpoints

| Method | Endpoint | Parameters | Response |
|---|---|---|---|
| `POST` | `/upload` | Form fields: `file` (binary), `reid_mode` (str, default `"fast"`), `line_mode` (str, default `"dual"`), `line_single_pos` (float, default `0.50`), `line_dual_a_pos` (float, default `0.45`), `line_dual_b_pos` (float, default `0.80`) | `UploadResponse` JSON + HTTP 200; HTTP 400 if not .mp4 |
| `GET` | `/status/{job_id}` | Path: `job_id` (str) | `StatusResponse` JSON; HTTP 404 if unknown |
| `GET` | `/result/{job_id}` | Path: `job_id` (str) | `ResultResponse` JSON if complete; HTTP 425 JSON `{status, progress}` if not ready; HTTP 404 if unknown |
| `GET` | `/video/{job_id}` | Path: `job_id` (str) | MP4 `FileResponse`; HTTP 404 if not complete or file missing |
| `GET` | `/report/{job_id}` | Path: `job_id` (str); query: `format` (str, `"csv"` or `"excel"`, default `"csv"`) | CSV or XLSX `FileResponse` with download filename header; HTTP 400 for invalid format; HTTP 404 if not complete |
| `GET` | `/static/{filename}` | Path: any filename in `outputs/` | Static file served from `outputs/` directory; supports byte-range requests |

### WebSocket

| Endpoint | Direction | Message shape |
|---|---|---|
| `WS /ws/{job_id}` | Server → client (every ~500 ms) | `{"status": str, "progress": float, "error": str \| null}` |
| `WS /ws/{job_id}` | Server → client (on missing job) | `{"error": "job not found"}` |

---

## 10. Configuration & Tunable Constants

All constants are defined at the top of `backend/tracker.py`.

| Constant | Default | Effect of changing it |
|---|---|---|
| `LINE_MODE` | `"single"` | Module-level fallback only; the actual mode is always passed by the caller via `VehicleTracker.__init__` |
| `LINE_SINGLE_POS` | `0.50` | Module-level fallback; used if the caller does not pass `line_single_pos` |
| `LINE_DUAL_A_POS` | `0.25` | Module-level fallback for Line A in dual mode |
| `LINE_DUAL_B_POS` | `0.75` | Module-level fallback for Line B in dual mode |
| `FLASH_FRAMES` | `10` | Number of frames the semi-transparent bbox fill is shown after a counting event; increase for a longer visual flash, decrease to shorten it |
| `MAX_TRACK_AGE` | `50` | Passed to ByteTrack as the number of frames to keep a lost track alive; increase to handle longer occlusions, decrease to free memory faster in dense scenes — currently only referenced in the constant declaration and not explicitly wired into the `model.track()` call, which uses ByteTrack's own `bytetrack.yaml` defaults |
| `REID_THRESHOLD` | `0.90` | Pearson correlation score above which a crossing is suppressed as a duplicate; raise to allow more re-counts (stricter match required), lower to suppress more aggressively |
| `FRAME_SKIP` | `1` | Process every Nth frame; `1` = every frame; `2` = every other frame; increase to trade accuracy for speed on slow hardware |

---

## 11. Output Files

All output files are written to `backend/outputs/` (relative to the directory from which `uvicorn` is started).

### `{job_id}_processed.mp4`

H.264 encoded video at the source video's original resolution and frame rate. Every frame from the source appears in the output (skipped frames are written as-is to preserve timing). On processed frames, the following are drawn:
- Coloured bounding box outline per detected vehicle (green=car, red=truck, blue=bus, yellow=motorcycle).
- Label text `"<class> #<track_id>"` above each box.
- Semi-transparent filled box over each newly counted vehicle for `FLASH_FRAMES` (10) frames.
- White horizontal counting line(s) spanning the full frame width.
- In single mode: `"Counted: <N>"` text above the line.
- In dual mode: `"Line A"` / `"Line B"` labels and `"Total Counted: <N>"` at bottom of frame.

### `{job_id}_report.csv`

Flat CSV with a header block followed by data rows. One row per unique vehicle crossing (not per frame).

Header block:
```
Traffic Analyzer Report
Job ID,<job_id>
Processing duration (sec),<float>
(blank row)
```

Data columns:
| Column | Content |
|---|---|
| `frame_index` | Frame number where the first crossing was detected for this track |
| `timestamp_sec` | `frame_index / fps`, 3 decimal places |
| `track_id` | ByteTrack integer ID |
| `vehicle_class` | `car` / `truck` / `bus` / `motorcycle` |
| `counter` | Sequential integer (1 = first vehicle counted, 2 = second, ...) |

### `{job_id}_report.xlsx`

Two-sheet workbook.

**Sheet `"Counted"`:** Same header block and columns as the CSV.

**Sheet `"Summary"`:** Two columns (`Metric`, `Value`). Rows:

| Metric | Value |
|---|---|
| Total unique vehicles | integer |
| Count by class: car | integer |
| Count by class: truck | integer |
| Count by class: bus | integer |
| Count by class: motorcycle | integer |
| Processing duration (sec) | float |
| ReID mode | "fast" / "accurate" |
| Line mode | "single" / "dual" |
| Video FPS | float |
| Video resolution | `<width>x<height>` |
| Total frames | integer |
| Processed frames | integer |

Both sheets have all columns auto-sized to the longest cell value.

---

## 12. Engineering Decisions

### YOLOv8n vs. larger models

**Chose:** YOLOv8n as the default ("fast" mode), with YOLOv8s as the "accurate" option.
**Considered:** YOLOv8m, YOLOv8l, YOLOv8x.
**Why:** Drone footage is typically a top-down view with small vehicles; YOLOv8n's speed is a larger practical advantage than the marginal recall improvement from bigger models. YOLOv8s at 1280px gives a meaningful jump in small-object detection without the latency penalty of the medium/large/extra-large models. Bigger models were excluded because local deployment assumes no GPU, and processing time grows non-linearly.

### ByteTrack vs. SORT vs. DeepSORT vs. StrongSORT

**Chose:** ByteTrack via the Ultralytics `tracker="bytetrack.yaml"` parameter.
**Considered:** SORT (simpler, no re-association of low-confidence boxes), DeepSORT (adds appearance embedding), StrongSORT (appearance + motion model improvements).
**Why:** ByteTrack associates both high-confidence and low-confidence detections in two passes, which handles partial occlusion better than SORT without requiring a separate appearance model network (as DeepSORT does). It is also the default tracker in Ultralytics and requires no additional model download. Appearance-based trackers (DeepSORT, StrongSORT) were excluded because their benefit over ByteTrack was deemed insufficient to justify the additional latency and dependency weight, especially given that HSV ReID is already handling re-entry suppression.

### Dual lines vs. single line

**Chose:** Dual as the default (UI default: Line A at 45%, Line B at 80%).
**Considered:** Single line only.
**Why:** A single central line misses vehicles that enter the bottom half of the frame and exit without crossing the line (e.g. slow-moving or turning vehicles). Two lines create a wider counting zone so a vehicle only needs to cross one of the two to be counted. Single mode remains available for simpler scenes or when the traffic corridor is narrow and predictable.

### HSV histogram vs. RGB for ReID

**Chose:** HSV (Hue × Saturation channels only; Value excluded).
**Considered:** RGB histogram, full HSV including Value.
**Why:** Hue encodes the actual colour of the vehicle and is relatively stable across lighting conditions. Saturation helps distinguish vivid colours from grey/white/black vehicles. Value is excluded because it varies with shadow, sun angle, and exposure changes in drone footage — including it would reduce match accuracy for the same vehicle observed in different lighting moments. RGB histograms conflate colour and brightness, making them sensitive to the same lighting variation problem.

### REID_THRESHOLD = 0.90

**Chose:** 0.90 (Pearson correlation).
**Considered:** Values from 0.75 to 0.98.
**Why:** The threshold was calibrated empirically on test footage. Below ~0.85 there are too many false suppression events (different vehicles with similar colours triggering matches). Above ~0.95 the check fails to catch genuine re-entries because natural histogram drift between frames lowers the score slightly. 0.90 represents a reasonable balance point on the available test data.

### Frame skip = 1 (every frame)

**Chose:** `FRAME_SKIP = 1`.
**Considered:** 2, 3 (every other/every third frame).
**Why:** Skipping frames improves throughput but risks missing a crossing if a vehicle is fast relative to the video frame rate. At `FRAME_SKIP = 1` every detection is tracked. The constant is exposed for users on slow hardware to increase. Skipped frames are still written to the output video (as raw, unannotated) to preserve timing.

### H.264 (`avc1`) codec over `mp4v`

**Chose:** `cv2.VideoWriter_fourcc(*"avc1")`.
**Considered:** `mp4v` (MPEG-4 Part 2).
**Why:** H.264 is the standard codec for browser-native inline video playback. `mp4v`-encoded files often require transcoding or a separate codec in the browser. H.264 output plays in the `<video>` element without any conversion. The `avc1` fourcc is available in the standard `opencv-python` binary wheel on Windows and macOS. On some Linux builds `mp4v` may be necessary as a fallback.

### In-memory `jobs{}` dict vs. Redis / database

**Chose:** Module-level Python dict in `main.py`.
**Considered:** Redis, SQLite, PostgreSQL.
**Why:** The stated scope is local single-session deployment. An in-memory dict requires no additional infrastructure, has zero latency, and is correct for a single uvicorn process with one concurrent user. The limitation (data lost on restart, no multi-process sharing) is documented. Adding a persistent store would require roughly swapping the dict reads/writes for a database client with minimal change to the surrounding logic.

### Static drone assumption

**Chose:** No camera-motion compensation.
**Considered:** Optical flow or homography-based stabilisation before tracking.
**Why:** The system is designed for footage from a hovering drone with minimal or no lateral movement. Camera motion would cause all track centroids to shift in the same direction between frames, potentially triggering false line crossings or breaking track associations. Stabilisation is not implemented; the system assumes the drone is static. Users should stabilise shaky footage in a pre-processing step if needed.

---

## 13. Known Limitations

- **In-memory job store:** All job state is lost if the backend server restarts. There is no job history or persistence.
- **Single process:** FastAPI runs as a single uvicorn process. Background processing uses FastAPI's `BackgroundTasks`, which shares the process with request handling. A long video will block other jobs from running concurrently.
- **Static drone only:** No camera motion compensation. Panning or vibrating drone footage will degrade tracking accuracy and may produce false line crossings.
- **Four vehicle classes only:** COCO classes 2 (car), 3 (motorcycle), 5 (bus), and 7 (truck). Pedestrians, cyclists, vans, and other vehicles are ignored.
- **One-directional counting logic:** A vehicle is counted the first time it crosses a line, regardless of direction. There is no directional split (e.g., inbound vs. outbound). In dual mode, a vehicle that passes through the scene from top to bottom will be counted once (first line crossed); it will not be double-counted because `crossed_ids` blocks subsequent crossings.
- **ReID uses colour only:** The HSV histogram does not encode vehicle shape or size. Two vehicles of the same colour (e.g., two identical white sedans) could suppress each other's counts if they cross near in time, since their histograms are near-identical.
- **`recently_counted` window of 50:** If more than 50 unique vehicles cross before a re-entry, the oldest histogram is evicted and that re-entry will not be caught by ReID.
- **`MAX_TRACK_AGE = 50` is declared but not explicitly passed:** The constant is defined in `tracker.py` but `model.track()` does not receive it as a parameter; ByteTrack uses whatever default is in `bytetrack.yaml`. Changing `MAX_TRACK_AGE` has no effect without also editing the ByteTrack config.
- **H.264 availability:** The `avc1` fourcc requires an H.264-capable OpenCV build. On some minimal Linux installs, `opencv-python` may produce a silent fallback to `mp4v`, which can cause browser playback failures. Check `cv2.getBuildInformation()` if the video player shows a blank screen.
- **No authentication or rate limiting:** The API has no auth layer. It is intended for local use only.
- **`deep-sort-realtime`, `scipy`, and `pandas` in `requirements.txt` are unused:** These are listed as dependencies but are not imported by any module in the current codebase. They appear to be remnants from an earlier implementation. Removing them from `requirements.txt` would shorten install time without any functional impact.
- **`api.js` hardcodes `http://localhost:8000`:** Deploying the frontend on any host other than localhost would require updating `baseURL` in `api.js` and the WebSocket URL in `connectProgressWS`.

---

## 14. Project Structure

```
smart-drone-analyzer/
│
├── backend/                        # Python FastAPI application; run uvicorn from here
│   ├── main.py                     # FastAPI app, route handlers, in-memory job store
│   ├── processor.py                # Outer pipeline loop: opens video, drives frame loop, builds summary
│   ├── tracker.py                  # VehicleTracker: YOLO+ByteTrack detection, line crossing, ReID, drawing
│   ├── reporter.py                 # CSV and Excel report generation from vehicle_log
│   ├── models.py                   # Pydantic request/response models
│   ├── requirements.txt            # Python dependencies
│   ├── __init__.py                 # Empty; marks backend/ as a Python package
│   ├── yolov8n.pt                  # YOLOv8n weights (downloaded by Ultralytics on first run)
│   ├── yolov8s.pt                  # YOLOv8s weights (downloaded by Ultralytics on first run)
│   ├── uploads/                    # Uploaded MP4 files; named <job_id>_<original_filename>
│   └── outputs/                    # Processed MP4s, CSV and Excel reports; named <job_id>_*
│
├── frontend/                       # React + Vite single-page application
│   ├── index.html                  # Vite entry HTML; mounts #root
│   ├── vite.config.js              # Vite config; dev server port 5173; /api proxy to :8000
│   ├── package.json                # Node dependencies and scripts
│   ├── package-lock.json           # Exact dependency lockfile
│   └── src/
│       ├── main.jsx                # ReactDOM.createRoot entry point
│       ├── App.jsx                 # BrowserRouter, top nav, three routes
│       ├── api.js                  # All backend communication: uploadVideo, getStatus, getResult, connectProgressWS
│       ├── index.css               # Global styles and CSS custom properties
│       └── components/
│           ├── Upload.jsx          # File picker, config sliders, line preview canvas, upload trigger
│           ├── Progress.jsx        # WebSocket progress bar and status display
│           └── Results.jsx         # Annotated video player, count stats, report download buttons
│
├── uploads/                        # Root-level uploads dir (created if uvicorn run from project root)
├── outputs/                        # Root-level outputs dir (created if uvicorn run from project root)
├── .gitignore                      # Ignores: uploads/, outputs/, node_modules/, __pycache__/, *.pyc, .env, dist/, .vite/
└── README.md                       # This file
```

> `uploads/` and `outputs/` (at both root and `backend/` level) are git-ignored and will not appear in a fresh clone. The backend creates them automatically on startup via `Path.mkdir(exist_ok=True)`. The effective directories are whichever are relative to the `uvicorn` working directory — run from `backend/` to keep files in `backend/uploads/` and `backend/outputs/`.
