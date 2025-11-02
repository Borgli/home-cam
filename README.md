# 🎥 Home Camera Surveillance System

AI-powered surveillance system with real-time object detection for Reolink NVR cameras.

## ✨ Features

- **Real-time Object Detection** - YOLOv11 with live bounding boxes
- **Multi-Camera Support** - Monitor up to 4 NVR channels simultaneously
- **Always-On Streams** - Cameras accessible on-demand via NVR
- **Detection Recording** - SQLite database for all detections
- **AI Chat Assistant** - Query surveillance data with natural language
- **Web Interface** - Beautiful neon-themed UI with live feeds
- **REST API** - Full FastAPI backend with automatic documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 20.19+ (for web UI)
- Reolink NVR with FLV streaming enabled

### Installation

1. **Clone and setup:**
```bash
cd E:\home-cam
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure your NVR:**
```bash
# Edit .env file
REOLINK_IP=192.168.2.112
REOLINK_USERNAME=admin
REOLINK_PASSWORD=your_password
YOLO_MODEL=yolo11n.pt
```

3. **Initialize cameras:**
```bash
python setup_nvr_cameras.py
```

4. **Start the backend:**
```bash
python run.py
```

5. **Access the system:**
- Web Viewer: http://localhost:8000/viewer
- API Docs: http://localhost:8000/docs
- Camera Streams: http://localhost:8000/cameras/nvr_ch1/stream

### Optional: Start Web UI

```bash
cd web
npm install
npm run dev
```

Access at http://localhost:5173

## 📹 Camera Access

Your cameras are **always available** through the NVR. No need to start/stop them!

**Direct streams:**
- Channel 1: `http://localhost:8000/cameras/nvr_ch1/stream`
- Channel 2: `http://localhost:8000/cameras/nvr_ch2/stream`
- Channel 3: `http://localhost:8000/cameras/nvr_ch3/stream`
- Channel 4: `http://localhost:8000/cameras/nvr_ch4/stream`

## 🛠️ Utilities

### Test System Health
```bash
python test_system.py
```

### Check Camera Status
```bash
python check_camera_status.py
```

### View Statistics
```bash
python get_statistics.py
```

### Test Streaming
```bash
python test_simple_stream.py
```

### Complete Diagnostic
```bash
python test_and_start.py
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/cameras` | GET | List all cameras |
| `/cameras/{id}/stream` | GET | MJPEG stream with detection |
| `/detections` | GET | Query detection history |
| `/statistics` | GET | Detection statistics |
| `/chat` | POST | AI chat about detections |
| `/viewer` | GET | Web interface |

Full API documentation: http://localhost:8000/docs

## 🏗️ Architecture

```
Reolink NVR (4 channels)
    ↓
Token-based Authentication (reolinkapi)
    ↓
FastAPI Backend (Python)
    ↓
YOLOv11 Detection (Ultralytics)
    ↓
SQLite Database
    ↓
WebSocket + REST API
    ↓
Web UI (SvelteKit) or Simple Viewer
```

## 🔧 Configuration

### Environment Variables (.env)

```env
# NVR Settings
REOLINK_IP=192.168.2.112
REOLINK_USERNAME=admin
REOLINK_PASSWORD=your_password
REOLINK_RTSP_PORT=554

# Detection Settings
YOLO_MODEL=yolo11n.pt
YOLO_CONFIDENCE=0.5
YOLO_DEVICE=cpu

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Optional: AI Chat
HF_TOKEN=your_huggingface_token
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.1
```

### YOLO Models

- `yolo11n.pt` - Nano (fastest, default)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large (most accurate)
- `yolo11x.pt` - Extra Large

## 📁 Project Structure

```
E:\home-cam\
├── main.py              # FastAPI application
├── run.py               # Startup script
├── config.py            # Configuration
├── database.py          # SQLAlchemy models
├── llm_service.py       # AI chat service
├── requirements.txt     # Python dependencies
├── .env                 # Configuration (not in git)
│
├── camera/
│   ├── __init__.py
│   ├── detector.py      # YOLOv11 wrapper
│   └── reolink_api.py   # NVR authentication
│
├── web/
│   ├── src/             # SvelteKit UI
│   └── static/          # Simple HTML viewer
│
└── utils/
    ├── test_system.py
    ├── check_camera_status.py
    └── setup_nvr_cameras.py
```

## 🎨 Web Interface

The system includes two UI options:

### 1. Simple HTML Viewer (No setup required)
- Access: `http://localhost:8000/viewer`
- Features: 4-camera grid, statistics, neon theme
- No dependencies: Pure HTML/CSS/JS

### 2. SvelteKit UI (Advanced)
- Access: `http://localhost:5173` (after `npm run dev`)
- Features: Full dashboard, WebSocket updates, chat interface
- Requires: Node.js 20.19+

## 🐛 Troubleshooting

### Backend won't start
```bash
# Check imports
python -c "import main"

# Check port
netstat -ano | findstr :8000

# View logs
python run.py
```

### Cameras not connecting
```bash
# Test NVR connection
ping 192.168.2.112

# Test authentication
python test_simple_stream.py

# Check credentials in .env
```

### No detections
```bash
# Lower confidence threshold
# Edit .env: YOLO_CONFIDENCE=0.3

# Check YOLO model
dir yolo11n.pt
```

## 📖 Documentation

- **README.md** - This file (complete guide)
- **QUICKSTART.md** - Step-by-step setup guide
- **QUICK_REFERENCE.md** - Command cheat sheet
- `.env.example` - Configuration template

## 🔐 Security Notes

For production deployment:
1. Change default passwords
2. Use PostgreSQL instead of SQLite
3. Enable HTTPS/SSL
4. Configure CORS properly
5. Add authentication
6. Use environment variables (not .env file)

## 🤝 Contributing

This is a personal project, but improvements are welcome!

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- **Ultralytics** - YOLOv11 object detection
- **FastAPI** - Modern Python web framework
- **SvelteKit** - Reactive web UI framework
- **reolinkapi** - Reolink camera integration
- **LangChain** - AI chat functionality

## 📞 Support

For issues:
1. Run `python test_and_start.py` for diagnostics
2. Check logs in terminal
3. Review API docs at `/docs`
4. Test individual components with utility scripts

---

**Built with:** FastAPI • SvelteKit • YOLOv11 • LangChain • SQLAlchemy

**Status:** ✅ Production Ready

**Last Updated:** November 1, 2025

