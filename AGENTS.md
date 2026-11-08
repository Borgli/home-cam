# AGENTS.md

This file explains the home-cam surveillance system for AI agents and developers who need to understand, set up, or extend the project.

## What This Project Is

A real-time multi-camera surveillance system that connects to a Reolink NVR, runs YOLO26 object detection on 4 camera feeds simultaneously, tracks objects with TrackTrack + Re-ID, persists detections to SQLite, and serves everything through a React dashboard.

## Repository Layout

```
home-cam/
├── batched_viewer.py            # THE MAIN FILE - Flask backend, camera streams,
│                                # YOLO detection, tracking, metrics, MJPEG serving
├── inference_backend.py         # Picks CUDA / OpenVINO / PyTorch-CPU per PC (benchmark + cache)
├── openvino_reid.py             # Runs the tracker's Re-ID encoder on OpenVINO where that is faster
├── reolink_native.py            # 4K main streams over Reolink's native protocol (port 9000), on demand
├── simple_viewer.py             # Simpler alternative (sequential detection, no tracking)
├── surveillance_tracker.yaml    # Default tracker config: TrackTrack + Re-ID, fixed cameras
├── botsort_tracker.yaml         # Alternative: BoT-SORT + Re-ID (TRACKER_CONFIG in .env)
├── model_cache/                 # OpenVINO exports + per-PC backend benchmark (auto-created, gitignored)
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
│   ├── vite.config.js           # Dev server on :5173, proxies /api/* and /video_feed/* to :5001 (BACKEND_URL)
│   ├── tailwind.config.js       # Dark neon theme (greens, cyans, magentas)
│   ├── start.cjs                # Node launcher: sets CWD, passes args to Vite, --backend <url> = proxy target
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
│       │   ├── camera/CameraCard.jsx      # Single camera with zoom (wheel + buttons), 4K view, 4K still
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
4. **Tracking**: When enabled, `model.track(persist=True)` runs the configured tracker (TrackTrack + Re-ID by default) per camera
5. **Smoothing**: `DetectionsSmoother` reduces bounding box jitter on tracked objects
6. **Annotation**: Bounding boxes, labels, and movement trails are drawn on the frame
7. **Streaming**: Annotated frames are JPEG-encoded and served as MJPEG multipart streams
8. **Persistence**: Every detection is inserted into SQLite (timestamp, channel, class, confidence, bbox, tracker_id)
9. **Frontend**: React app polls `/api/metrics` and `/get_config` every 2 seconds, displays MJPEG via `<img>` tags
10. **4K view** (on demand): the dashboard can switch a camera to its 3840x2160 main stream; see [4K Streams](#4k-streams)

### Threading Model (batched_viewer.py)

- **Main thread**: Flask HTTP server
- **4 frame capture threads**: One per camera, continuously read from NVR
- **1 batch detection thread**: Collects frames, runs YOLO, annotates, stores results
- **N stream generator threads**: One per connected browser per camera (MJPEG)
- **Up to 2 4K stream threads** (`reolink_native.NativeStream`): started from the dashboard, each reads and decodes one camera's 4K stream; a reaper thread stops streams nobody has requested for `REOLINK_HD_IDLE_SECONDS`
- Thread safety via locks: `frames_lock`, `annotated_lock`, `detection_lock`, `config_lock`, `metrics_lock`, `hd_overlay_lock`

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

To run a second copy next to a running one (e.g. to test a change without stopping it): `python batched_viewer.py --port 5002` and `node frontend/start.cjs --port 5174 --strictPort --backend http://localhost:5002`, then open http://localhost:5174. Both copies share the NVR's limit of two 4K streams.

### Environment Variables

All config is in `.env`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `REOLINK_IP` | `192.168.2.112` | NVR IP address |
| `REOLINK_USERNAME` | `admin` | NVR login |
| `REOLINK_PASSWORD` | (required) | NVR password |
| `YOLO_MODEL` | `yolo26n.pt` | Detection model (downloaded automatically) |
| `YOLO_DEVICE` | `auto` | `auto` picks the fastest backend on this PC (see below); or force `cpu`, `cuda` / `cuda:N` / `N` (NVIDIA GPU N), `intel:gpu`, `intel:cpu`, `intel:npu` |
| `YOLO_REBENCHMARK` | `0` | `1` re-exports the OpenVINO models and re-runs the backend benchmark on the next start |
| `TRACKER_CONFIG` | `surveillance_tracker.yaml` | Tracker config file (`botsort_tracker.yaml` for BoT-SORT) |
| `YOLO_CONFIDENCE` | `0.5` | Minimum detection confidence |
| `REOLINK_MEDIA_PORT` | `9000` | NVR port for Reolink's native protocol (4K view) |
| `REOLINK_HD_MAX_STREAMS` | `2` | Cameras that may stream 4K at once; the RLN8-410-E refuses a third |
| `REOLINK_HD_IDLE_SECONDS` | `20` | Stop a camera's 4K stream after this long without requests |
| `REOLINK_HD_DECODER` | `auto` | 4K decoder: `auto` (GPU if available), `software`, or an FFmpeg device (`d3d11va`, `cuda`, `vaapi`, `qsv`) |
| `BACKEND_PORT` | `5001` | Backend port (`--port N` overrides) |

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

`TRACKER_CONFIG` in `.env` picks the tracker config file (default `surveillance_tracker.yaml`):
- `surveillance_tracker.yaml`: **TrackTrack + Re-ID** (default), tuned for fixed cameras
- `botsort_tracker.yaml`: BoT-SORT + Re-ID (the previous default)

Key options:
- `tracker_type`: `tracktrack`, `botsort`, `bytetrack`, `ocsort`, `deepocsort`, `fasttrack` (Ultralytics ≥ 8.4.63)
- `with_reid`: `true`/`false` - appearance-based re-identification (TrackTrack, BoT-SORT, Deep OC-SORT)
- `track_buffer`: frames to remember lost tracks (higher = more persistent IDs)
- `track_high_thresh` / `new_track_thresh`: keep at or below the app's 0.5 confidence filter, or confident detections go untracked
- `gmc_method`: `none` for fixed cameras, `sparseOptFlow` for moving cameras

Why TrackTrack (Nov 2026): all six Ultralytics trackers were run on the same 90 s recording from all 4 cameras (night, mostly static scene), with the same OpenVINO detector and synthetic occlusions. TrackTrack with thresholds at 0.5 created 4 IDs vs 10 for BoT-SORT + Re-ID, re-labelled no objects, tracked 99.9% of confident detections, and cost ~36 vs ~78 ms per frame. Every tracker recovered all synthetic occlusions, and the footage had no people or cars moving, so check behaviour in daylight with motion before tuning further. With Ultralytics' default thresholds (0.6/0.7), TrackTrack left ~10% of confident detections untracked.

### Changing the YOLO Model

Set `YOLO_MODEL=yolo26s.pt` (or m/l/x) in `.env`. The model downloads automatically on first run. Larger models are more accurate but slower, especially on CPU. With `YOLO_DEVICE=auto`, a new model triggers a one-time OpenVINO export and benchmark on Intel PCs.

### Inference Backends (running on different PCs)

`inference_backend.py` decides how YOLO runs, once per PC, when `YOLO_DEVICE=auto`:

1. **NVIDIA GPU** visible to PyTorch → PyTorch on CUDA (or `<model>_batch4.engine` if a TensorRT engine exists). No benchmark.
2. **OpenVINO installed** → exports the model to OpenVINO IR (batch 1 and batch 4, FP16) into `model_cache/<name>-<weights hash>_b<N>_openvino_model/`, then times PyTorch-CPU against OpenVINO on every Intel device (iGPU/Arc GPU, CPU, NPU) on a 4-frame batch. OpenVINO is used only if it is at least 15% faster (`MIN_SPEEDUP`).
3. Otherwise → PyTorch on the CPU.

Exports and the cached decision are keyed by a hash of the weights file's contents, so replacing the weights (e.g. a retrained `best.pt`) or pointing `YOLO_MODEL` at another file with the same name exports and benchmarks again instead of reusing a stale model. The decision is cached in `model_cache/backend_choice.json`, keyed by hostname, hardware, library versions and that weights hash, so the same checkout works across PCs. OpenVINO models have a fixed batch size: `BACKEND.predict_batch()` pads partial batches (e.g. a camera offline), and sequential mode and tracking use the batch-1 export. The active backend is exposed as `inference_backend` in `GET /get_config` and shown in the dashboard header.

**Tracking Re-ID.** With YOLO26 (end-to-end head), the trackers' `model: auto` can't reuse detector features, so Ultralytics runs a separate `yolo26n-cls.pt` encoder on a crop of every detected object. The same one-time benchmark times that encoder (PyTorch vs OpenVINO per device, for 1 and 4 objects). Where OpenVINO is at least 15% faster, `BACKEND.tracker_config()`:
1. exports the classifier backbone + average pooling to `model_cache/yolo26n-cls_reid_openvino_model/` (same embedding as Ultralytics' path, cosine similarity 1.0),
2. hooks Ultralytics' `ReID` class (`ultralytics.trackers.utils.reid`, or `bot_sort` in older versions) so that model path returns `openvino_reid.OpenVINOReID` on the benchmarked device,
3. writes `model_cache/tracker_openvino_reid.yaml`: a copy of the selected tracker config (`TRACKER_CONFIG`) with `model:` pointing at the export. Edit the source config, never the generated file. This applies to TrackTrack, BoT-SORT and Deep OC-SORT configs.

If the hook point is missing, the config sets an explicit `model:`, or anything fails, tracking keeps the source config and Ultralytics' default Re-ID. A hook is used instead of only pointing `model:` at the export because Ultralytics would then run the encoder on the detector's PyTorch device and couldn't target the benchmarked OpenVINO device.

Ultralytics also ships dedicated Re-ID models `yolo26{n,s,m,l,x}-reid.onnx` (448 px), not used here. Setting one as `model:` bypasses the OpenVINO encoder and runs it through Ultralytics directly.

## 4K Streams

The cameras' main streams are 3840x2160 **H.265** (configured 20 fps / 8 Mbit/s; about 12 fps and 5-6 Mbit/s at night). Detection keeps using the NVR's 1280x720 H.264 `ext` FLV stream. The 4K stream feeds the dashboard's 4K view and full-resolution stills.

**Why a native-protocol client.** The NVR (RLN8-410-E, hardware H3MB16, firmware v2.0.0.280, the last release for that hardware) can't relay H.265 over its standard protocols (tested Nov 2026):
- FLV/RTMP `channelN_main`: the FLV header announces HEVC (codec id 12), then no video arrives.
- RTSP `h264Preview_0N_main` (what ONVIF advertises, labelled H264): the NVR never answers DESCRIBE. `h265Preview_0N_main` and `Preview_0N_main` return 404. There is no ONVIF Profile T.
- Reolink's native "Baichuan" protocol on the media port (9000), which Reolink's own apps use, does deliver it. `reolink_native.py` implements the minimum: legacy login (for the nonce and encryption choice), the XOR-encoded modern login, AES-128-CFB afterwards (`cryptography` if installed, else OpenSSL's libcrypto through ctypes; Python ships it on Windows), `Preview` (message 3) and the BcMedia packet format. The details follow the open-source neolink and reolink_aio projects.

**Limit: two 4K streams at once.** The NVR answers a third concurrent preview with code 400 (measured while the app also pulled its four 720p streams). `HDStreamManager` enforces `REOLINK_HD_MAX_STREAMS`, starts streams on demand and stops one after `REOLINK_HD_IDLE_SECONDS` without requests. Another client watching in 4K (e.g. the Reolink app in "Clear" mode) takes a slot too; the stream then reports `refused` and retries.

**Decoding cost.** PyAV decodes H.265 on the GPU (`REOLINK_HD_DECODER=auto`: D3D11VA on Windows, CUDA/VAAPI on Linux) and falls back to software. On the i7-1265U / Iris Xe laptop that is ~6.6 ms of CPU per 4K frame with D3D11VA vs ~24 ms in software. Converting a whole 4K frame to BGR costs ~40 ms, so `frame_to_bgr()` crops in YUV first, then scales and converts with OpenCV (BT.709/BT.601, full or limited range from the frame's tags): a 2x-zoomed 1280x720 view costs ~9 ms and the unzoomed view ~16 ms on one core.

**Endpoints** (under `/api`, so the Vite proxy forwards them):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/hd/<ch>` | POST | Start (or keep) the camera's 4K stream and wait for its first frame. 409 = limit reached, 502 = NVR refused or unreachable |
| `/api/hd/<ch>` | DELETE | Stop it, freeing the NVR slot |
| `/api/hd/<ch>/frame?x0=&y0=&x1=&y1=&w=` | GET | JPEG of the region x0..x1, y0..y1 (0-1, may extend past the edges) of the newest 4K frame, `w` px wide, with the 720p detector's boxes drawn (`overlay=0` to skip) and privacy zones blacked out. 404 = not started, 503 = reconnecting |
| `/api/hd/<ch>/snapshot` | GET | Full-resolution JPEG from the running 4K stream, else the NVR's HTTP `cmd=Snap` (~1.6 s); privacy zones blacked out |
| `/api/hd/status` | GET | Running streams (status, fps, bitrate, decoder) and the limit; also in `GET /get_config` as `hd` |

**Dashboard.** Each camera card has a **4K** toggle and a still button. With 4K on, the card polls `/frame` for the region it shows (at most ~12 times a second), so zooming in shows real 4K detail; the 720p stream stays underneath while a new region loads. The toggle lives in the store, so it survives expand/collapse, and if the backend stopped an idle stream the card starts it again once.

**Privacy.** 4K frames and stills always have the privacy zones blacked out, even with detection off (the 720p stream only masks them while detection runs).

**Debugging.** `venv\Scripts\python reolink_native.py <channel> [seconds]` streams one camera, prints its status every 2 s and saves a 4K still. A 4-minute run showed no drops without keepalive messages.

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
- **Tracking is slow on CPU**: Tracking uses per-camera inference (not batched). Keep `YOLO_DEVICE=auto` so a CUDA or Intel GPU is used when available
- **Backend choice looks wrong after a driver or hardware change**: set `YOLO_REBENCHMARK=1` (or delete `model_cache/backend_choice.json`) and restart
- **Privacy zones not blocking detections**: Backend must be restarted after the `backend_api` code was added. Zones are loaded from SQLite every 2 seconds
- **"At most 2 cameras can stream 4K at once"**: the NVR's limit. Turn 4K off on another camera; a backend copy on another port or the Reolink app in "Clear" mode also uses slots
- **4K says "refused" or "login failed"**: check that the NVR's media port (`REOLINK_MEDIA_PORT`, 9000) is reachable and the `.env` credentials are right; the backend log shows `[4K]` lines per camera
