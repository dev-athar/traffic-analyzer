# Smart Drone Traffic Analyzer

Phase 1 scaffold — upload pipeline only. CV logic is not yet implemented.

## Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: React 18 + Vite

## Quick Start

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:5173  
Backend API runs on http://localhost:8000  
All frontend `/api/*` calls proxy to the backend automatically.

## Test Upload

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@test.mp4" \
  -F "reid_mode=fast"
```

## Roadmap

- **Phase 1** (current): Scaffold + file upload
- **Phase 2**: YOLO detection + DeepSORT tracking + WebSocket progress
- **Phase 3**: ReID, analytics aggregation, Excel report, annotated video export
