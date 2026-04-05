# AGENTS.md

This file explains the home-cam surveillance system for AI agents and developers who need to understand, set up, or extend the project.

## What This Project Is

A real-time multi-camera surveillance system that connects to a Reolink NVR, runs YOLO26 object detection on 4 camera feeds simultaneously, tracks objects with BoT-SORT + Re-ID, persists detections to SQLite, and serves everything through a React dashboard.

## Repository Layout

```
home-cam/
├── batched_viewer.py            # THE MAIN FILE - Flask backend, camera streams,
│                                # YOLO detection, tracking, metrics, MJPEG serving
├── simple_viewer.py             # Simpler alternative (sequential detection, no tracking)
├── surveillance_tracker.yaml    # BoT-SORT tracker config (Re-ID, fixed cameras)
├── requirements.txt             # Python deps (flask, ultralytics, supervision, etc.)
├── .env                         # Runtime config - NVR credentials, model, device
├── .env.example                 # Template for .env (safe to commit)
│
├── backend_api/                 # Flask Blueprints extending batched_viewer.py
│   ├── __init__.py              # register_blueprints(app) - wires all blueprints
│   ├── database.py              # SQLite schema, insert/query helpers, schema info
│   ├── zones.py                 # CRUD for privacy zones, counting zones, crossing lines
│   ├── events_api.py            # Paginated event log, stats, SQL query execution
│   ├── classes_api.py           # COCO class listing and filter (enable/disable classes)
│   └── llm_service.py           # Ollama + Gemma 4 E2B: natural language -> SQL -> results
│
├── frontend/                    # React 18 + Vite + Tailwind CSS SPA
│   ├── package.json             # npm deps: react, zustand, recharts, konva, lucide
│   ├── vite.config.js           # Dev server on :5173, proxies /api/* and /video_feed/* to :5001
│   ├── tailwind.config.js       # Dark neon theme (greens, cyans, magentas)
│   ├── start.cjs                # Node launcher that sets CWD before running Vite
│   └── src/
│       ├── App.jsx              # React Router + layout + 2s polling for config/metrics
│       ├── api/client.js        # Fetch wrapper for all backend endpoints
│       ├── stores/store.js      # Zustand store: config, zones, events, classes, UI state
│       ├── pages/
│       │   ├── Dashboard.jsx    # 2x2 camera grid, drawing tools, event log
│       │   ├── Analytics.jsx    # Recharts: FPS, batch time, efficiency, confidence
│       │   ├── Database.jsx     # Browse detections, SQL editor, AI query (3 tabs)
│       │   └── Settings.jsx     # Toggle cards, FPS slider, 80-class selector, zone list
│       ├── components/
│       │   ├── camera/CameraCard.jsx      # Single camera with zoom (wheel + buttons)
│       │   ├── camera/CameraOverlay.jsx   # Konva canvas for drawing zones/lines
│       │   ├── drawing/DrawingToolbar.jsx # Tool selector: pointer/privacy/counting/line
│       │   ├── events/EventLog.jsx        # Collapsible bottom panel, real-time detections
│       │   ├── database/TableViewer.jsx   # Paginated table with CSV export
│       │   ├── database/QueryEditor.jsx   # SQL textarea with Ctrl+Enter execution
│       │   ├── database/LLMQuery.jsx      # Natural language input + example queries
│       │   ├── settings/ClassSelector.jsx # 80 COCO classes grid with search
│       │   ├── settings/FPSControl.jsx    # Range slider 1-60
│       │   └── layout/Sidebar.jsx, Header.jsx
│       └── hooks/usePolling.js            # Generic setInterval polling hook
│
├── .claude/launch.json          # Dev server configs for Claude Code preview tool
└── surveillance.db              # SQLite database (auto-created, gitignored)
```

## How the System Works

### Data Flow

1. **Frame Capture**: 4 threads continuously read FLV streams from the Reolink NVR via HTTP token auth
2. **Privacy Masking**: Before detection, privacy zone regions (from SQLite) are blacked out on the frame
3. **Detection**: YOLO26 runs on the (masked) frames - batched (4 at once) or per-camera when tracking
4. **Tracking**: When enabled, `model.track(persist=True)` runs BoT-SORT with Re-ID per camera
5. **Smoothing**: `DetectionsSmoother` reduces bounding box jitter on tracked objects
6. **Annotation**: Bounding boxes, labels, and movement trails are drawn on the frame
7. **Streaming**: Annotated frames are JPEG-encoded and served as MJPEG multipart streams
8. **Persistence**: Every detection is inserted into SQLite (timestamp, channel, class, confidence, bbox, tracker_id)
9. **Frontend**: React app polls `/api/metrics` and `/get_config` every 2 seconds, displays MJPEG via `<img>` tags

### Threading Model (batched_viewer.py)

- **Main thread**: Flask HTTP server
- **4 frame capture threads**: One per camera, continuously read from NVR
- **1 batch detection thread**: Collects frames, runs YOLO, annotates, stores results
- **N stream generator threads**: One per connected browser per camera (MJPEG)
- Thread safety via locks: `frames_lock`, `annotated_lock`, `detection_lock`, `config_lock`, `metrics_lock`

### Key Global State

```python
DETECTION_ENABLED = False    # Toggle via POST /toggle_detection
TRACKING_ENABLED = False     # Toggle via POST /toggle_tracking
BATCHED_MODE = True          # Toggle via POST /toggle_batched_mode
AUTO_FPS = True              # Toggle via POST /toggle_auto_fps
TARGET_FPS = 12              # Set via POST /set_target_fps
```

## Setup Instructions

### Prerequisites

- Python 3.10+ with pip
- Node.js 20+ with npm
- A Reolink NVR accessible on the local network
- (Optional) Ollama for AI database queries

### First-Time Setup

```bash
# 1. Create Python virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 2. Install Python dependencies
pip install -r requirements.txt
# Core: flask, flask-cors, ultralytics, supervision, opencv-python, python-dotenv, reolinkapi

# 3. Create .env from template
cp .env.example .env
# Edit .env with your NVR IP, username, password

# 4. Install frontend dependencies
cd frontend
npm install
cd ..

# 5. (Optional) Install Ollama + Gemma 4 for AI queries
# Download from https://ollama.com then:
# ollama pull gemma4:e2b
```

### Running

```bash
# Backend (port 5001)
venv\Scripts\activate
python batched_viewer.py

# Frontend (port 5173) - in a separate terminal
cd frontend
npm run dev
```

The frontend proxies all API calls to localhost:5001 via Vite's proxy config.

### Environment Variables

All config is in `.env`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `REOLINK_IP` | `192.168.2.112` | NVR IP address |
| `REOLINK_USERNAME` | `admin` | NVR login |
| `REOLINK_PASSWORD` | (required) | NVR password |
| `YOLO_MODEL` | `yolo26n.pt` | Detection model (downloaded automatically) |
| `YOLO_DEVICE` | `cpu` | `cpu` or `cuda` for GPU |
| `YOLO_CONFIDENCE` | `0.5` | Minimum detection confidence |

**Do not commit `.env`** - it contains NVR credentials. It is gitignored.

## How to Extend This Project

### Adding a New Backend API Endpoint

1. Create a route in the appropriate `backend_api/*.py` file (or create a new Blueprint)
2. If new Blueprint, register it in `backend_api/__init__.py`
3. Add the corresponding fetch call in `frontend/src/api/client.js`
4. The Vite proxy already forwards all `/api/*` requests to the backend

### Adding a New Frontend Page

1. Create `frontend/src/pages/NewPage.jsx`
2. Add a route in `frontend/src/App.jsx`: `<Route path="/newpage" element={<NewPage />} />`
3. Add nav link in `frontend/src/components/layout/Sidebar.jsx`

### Adding a New Detection Feature

Detection happens in `batched_viewer.py` in `batch_detection_thread()`. The flow is:
1. Frames collected from `latest_frames` dict
2. Privacy zones applied (line ~270)
3. YOLO inference (line ~280)
4. Post-processing loop per camera (line ~305+):
   - `sv.Detections.from_ultralytics(result)` converts results
   - Confidence filtering
   - Smoothing (if tracking)
   - Metrics recording + DB persistence
   - Annotation (box, trace, label)
   - Store in `annotated_frames[channel]`

### Adding New Zone Types

1. Add the type string to the frontend `DrawingToolbar.jsx` and `CameraOverlay.jsx`
2. The backend `zones.py` already supports arbitrary `type` values in the `zones` table
3. Add processing logic in `batch_detection_thread()` to act on the zone (e.g., counting objects inside a polygon)

### Changing the Tracker

Edit `surveillance_tracker.yaml`. Key options:
- `tracker_type`: `botsort` or `bytetrack`
- `with_reid`: `true`/`false` - appearance-based re-identification
- `track_buffer`: Frames to remember lost tracks (higher = more persistent IDs)
- `gmc_method`: `none` for fixed cameras, `sparseOptFlow` for moving cameras

### Changing the YOLO Model

Set `YOLO_MODEL=yolo26s.pt` (or m/l/x) in `.env`. The model downloads automatically on first run. Larger models are more accurate but slower, especially on CPU.

## Database

SQLite at `surveillance.db` (auto-created). Three tables:

- **detections**: Every detected object with timestamp, channel, class, confidence, bounding box, tracker_id
- **zones**: Privacy zones, counting zones, crossing lines with coordinates (stored as JSON)
- **zone_events**: Events triggered by zones (entry, exit, cross)

The `backend_api/database.py` module provides `insert_detections_batch()`, `query_db()`, and `get_schema_info()`.

SQL queries from the frontend are restricted to SELECT only for safety.

## Conventions

- **Backend**: Flask with inline HTML for legacy pages, Blueprint pattern for new API endpoints
- **Frontend**: React functional components, Zustand for state, Tailwind for styling
- **Theme**: Dark background (#0a0a0f), neon accents (green #00ff88, cyan #00ccff, magenta #ff00ff)
- **API style**: JSON REST, no authentication (local network only)
- **Config**: All runtime toggles via POST endpoints, persistent config in `.env`

## Common Issues

- **"REOLINK_PASSWORD must be set"**: Create `.env` from `.env.example` and set your NVR password
- **Model download slow**: YOLO models (~5MB for nano) download from GitHub on first run
- **Frontend can't reach backend**: Ensure backend is running on port 5001 before starting frontend
- **Tracking is slow on CPU**: Tracking uses per-camera inference (not batched). Consider `YOLO_DEVICE=cuda` if GPU available
- **Privacy zones not blocking detections**: Backend must be restarted after the `backend_api` code was added. Zones are loaded from SQLite every 2 seconds
