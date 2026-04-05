# Home Camera Surveillance System

AI-powered surveillance system with real-time object detection, tracking, and a modern React dashboard for Reolink NVR cameras.

## Features

- **YOLO26 Object Detection** - Real-time detection with 80 COCO classes on live camera feeds
- **BoT-SORT Tracking with Re-ID** - Appearance-based re-identification so tracked objects keep their IDs even after temporary occlusion
- **4-Camera Grid** - Monitor up to 4 NVR channels simultaneously with zoom and fullscreen
- **Drawing Tools** - Privacy zones, counting zones, and crossing lines drawn directly on camera feeds
- **Privacy Zone Masking** - Drawn privacy zones are applied server-side before detection, truly hiding regions from YOLO
- **Movement Trails** - TraceAnnotator draws motion paths behind tracked objects
- **Detection Smoothing** - DetectionsSmoother reduces bounding box jitter
- **SQLite Persistence** - All detections stored with timestamps, bounding boxes, class, confidence, and tracker IDs
- **SQL Query Editor** - Write and execute SELECT queries against the detection database
- **AI-Powered Queries** - Natural language to SQL via Ollama + Gemma 4 E2B (local, private)
- **Real-time Event Log** - Scrolling detection feed with confidence bars
- **Analytics Dashboard** - Recharts-based FPS, batch time, efficiency, and confidence charts
- **Batched Inference** - Process all 4 cameras in a single GPU forward pass
- **Performance Comparison** - Batch vs sequential metrics with optimization recommendations

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 20+ (for the React frontend)
- Reolink NVR with FLV streaming enabled

### Installation

```bash
git clone <repo-url>
cd home-cam

# Python backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# React frontend
cd frontend
npm install
cd ..
```

### Configuration

Copy `.env.example` to `.env` and edit:

```env
REOLINK_IP=192.168.2.112
REOLINK_USERNAME=admin
REOLINK_PASSWORD=your_password
YOLO_MODEL=yolo26n.pt
YOLO_DEVICE=cpu               # or cuda for GPU
```

### Running

```bash
# Terminal 1: Start backend (cameras + detection + API)
venv\Scripts\activate
python batched_viewer.py
# Runs on http://localhost:5001

# Terminal 2: Start frontend
cd frontend
npm run dev
# Runs on http://localhost:5173 (proxies API calls to backend)
```

Open http://localhost:5173 in your browser.

### Optional: AI Queries with Ollama

To use natural language database queries:

```bash
# Install Ollama from https://ollama.com
ollama pull gemma4:e2b
# Ollama runs automatically on localhost:11434
```

## Architecture

```
Reolink NVR (4 channels via FLV)
    |
    v
Frame Capture Threads (1 per camera)
    |
    v
Privacy Zone Masking (black out regions before detection)
    |
    v
YOLO26 Detection (batched or per-camera with tracking)
    |--- BoT-SORT + Re-ID tracking (persist=True per camera)
    |--- DetectionsSmoother (reduces box jitter)
    |--- TraceAnnotator (movement trails)
    |
    v
MJPEG Streams + SQLite Persistence
    |
    v
Flask Backend (REST API + MJPEG)     React Frontend (Vite + Tailwind)
    port 5001                             port 5173
```

## Project Structure

```
home-cam/
├── batched_viewer.py            # Main backend: cameras, detection, tracking, API
├── simple_viewer.py             # Simpler single-process viewer (no tracking)
├── surveillance_tracker.yaml    # BoT-SORT + Re-ID tracker config
├── requirements.txt             # Python dependencies
├── .env                         # Configuration (not in git)
├── .env.example                 # Configuration template
│
├── backend_api/                 # Flask Blueprint extensions
│   ├── __init__.py              # Blueprint registration
│   ├── database.py              # SQLite schema, queries, persistence
│   ├── zones.py                 # Zone/line CRUD endpoints
│   ├── events_api.py            # Event log, SQL query, stats endpoints
│   ├── classes_api.py           # COCO class filter endpoints
│   └── llm_service.py           # Ollama/Gemma 4 text-to-SQL integration
│
└── frontend/                    # React SPA
    ├── package.json
    ├── vite.config.js           # Dev server with API proxy to backend
    ├── tailwind.config.js       # Dark neon theme config
    └── src/
        ├── App.jsx              # Router + layout + polling
        ├── api/client.js        # API client for all backend calls
        ├── stores/store.js      # Zustand state (config, zones, events)
        ├── pages/
        │   ├── Dashboard.jsx    # Camera grid + drawing tools + event log
        │   ├── Analytics.jsx    # Recharts performance dashboard
        │   ├── Database.jsx     # Browse, SQL query, AI query tabs
        │   └── Settings.jsx     # Toggles, FPS slider, class selector, zones
        └── components/
            ├── camera/          # CameraCard, CameraOverlay (Konva canvas), zoom
            ├── drawing/         # DrawingToolbar (privacy/counting/line tools)
            ├── events/          # EventLog (real-time detection feed)
            ├── database/        # TableViewer, QueryEditor, LLMQuery
            ├── settings/        # ClassSelector (80 COCO classes), FPSControl
            └── layout/          # Sidebar, Header with status pills
```

## API Endpoints

### Existing (batched_viewer.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Legacy 4-camera HTML view |
| `/video_feed/<channel>` | GET | MJPEG stream (0-3) |
| `/toggle_detection` | POST | Toggle YOLO detection on/off |
| `/toggle_tracking` | POST | Toggle BoT-SORT + Re-ID tracking |
| `/toggle_batched_mode` | POST | Toggle batched vs sequential inference |
| `/toggle_auto_fps` | POST | Toggle automatic FPS detection |
| `/set_target_fps` | POST | Set manual target FPS (1-60) |
| `/get_config` | GET | Current configuration state |
| `/api/metrics` | GET | Full metrics (FPS, detections, batch times) |
| `/api/batch_stats` | GET | Quick batch statistics |
| `/api/comparison_metrics` | GET | Batch vs sequential comparison |
| `/analytics` | GET | Legacy analytics page |
| `/comparison` | GET | Legacy comparison page |

### New (backend_api/)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/zones` | GET/POST | List or create zones (privacy, counting, crossing line) |
| `/api/zones/<id>` | PUT/DELETE | Update or delete a zone |
| `/api/classes` | GET | List 80 COCO classes with enabled status |
| `/api/classes/filter` | POST | Set which classes to detect |
| `/api/events` | GET | Paginated detection event log |
| `/api/events/stats` | GET | Detection statistics (per class, per camera) |
| `/api/db/query` | POST | Execute a read-only SQL query |
| `/api/db/llm-query` | POST | Natural language to SQL via Ollama |
| `/api/db/schema` | GET | Database schema (for LLM context) |

## YOLO Models

The default is `yolo26n.pt` (YOLO26 nano), optimized for CPU:

| Model | mAP | CPU Speed | Parameters |
|-------|-----|-----------|------------|
| `yolo26n.pt` | 40.9 | 38.9ms | 2.4M |
| `yolo26s.pt` | 47.0 | — | 9.6M |
| `yolo26m.pt` | 51.5 | — | 21.8M |
| `yolo26l.pt` | 53.2 | — | 44.5M |

Set via `YOLO_MODEL` in `.env`. Models are downloaded automatically on first run.

## Tracker Configuration

The `surveillance_tracker.yaml` configures BoT-SORT for fixed surveillance cameras:

- **with_reid: true** - Appearance-based re-identification
- **model: auto** - Reuses YOLO backbone features (zero extra overhead)
- **gmc_method: none** - No motion compensation (cameras are fixed)
- **track_buffer: 90** - Keep lost tracks for ~8 seconds

## Database Schema

Detections are persisted to `surveillance.db` (SQLite):

```sql
detections (id, timestamp, channel, class_name, class_id, confidence, x1, y1, x2, y2, tracker_id)
zones (id, camera, type, label, coords, classes, color, enabled, created_at)
zone_events (id, timestamp, zone_id, event_type, class_name, tracker_id)
```

## Security Notes

- NVR credentials are stored in `.env` (gitignored)
- SQL queries via the frontend are restricted to SELECT only
- The LLM query endpoint validates generated SQL before execution
- No secrets are hardcoded in source files

## License

MIT License - See LICENSE file

## Acknowledgments

- [Ultralytics](https://ultralytics.com) - YOLO26 object detection
- [Supervision](https://supervision.roboflow.com) - Annotation, tracking utilities, smoothing
- [Ollama](https://ollama.com) - Local LLM runtime
- [reolinkapi](https://github.com/ReolinkCameraAPI/reolinkapipy) - NVR integration

**Built with:** Flask, React, Vite, Tailwind CSS, YOLO26, BoT-SORT, Recharts, Konva, Zustand, SQLite, Ollama
