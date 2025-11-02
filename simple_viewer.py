"""
4-Channel Camera Viewer - All NVR cameras in a grid
Displays all 4 channels simultaneously with optional YOLOv11 detection
"""
from flask import Flask, Response, request, jsonify
from reolinkapi import Camera
import cv2
import threading
import time
import os
from dotenv import load_dotenv
from ultralytics import YOLO
import supervision as sv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# NVR Configuration from environment variables
NVR_IP = os.getenv('REOLINK_IP', '192.168.2.112')
USERNAME = os.getenv('REOLINK_USERNAME', 'admin')
PASSWORD = os.getenv('REOLINK_PASSWORD')

if not PASSWORD:
    raise ValueError("REOLINK_PASSWORD must be set in .env file")

# Detection configuration
DETECTION_ENABLED = False  # Global toggle for detection
detection_lock = threading.Lock()

# Performance metrics tracking
performance_metrics = {
    'fps': {i: [] for i in range(4)},  # FPS per camera
    'detections': [],  # Detection events with timestamp
    'confidence': [],  # Confidence scores
    'processing_time': {i: [] for i in range(4)},  # Detection processing time per camera
    'frames_skipped': {i: 0 for i in range(4)},  # Total frames skipped per camera
    'frames_processed': {i: 0 for i in range(4)},  # Total frames processed per camera
    'total_detections': 0,
    'start_time': None
}
metrics_lock = threading.Lock()

# Initialize YOLO model with configuration from environment
YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolo11n.pt')
YOLO_DEVICE = os.getenv('YOLO_DEVICE', 'cpu')

print(f"\n[INFO] Loading {YOLO_MODEL} on {YOLO_DEVICE.upper()}...")
try:
    model = YOLO(YOLO_MODEL)
    if YOLO_DEVICE.lower() == 'cuda' or YOLO_DEVICE.startswith('0'):
        model.to('cuda')
        print(f"[OK] {YOLO_MODEL} loaded on GPU")
    else:
        print(f"[OK] {YOLO_MODEL} loaded on CPU")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    # Fallback to nano model on CPU
    model = YOLO("yolo11n.pt")
    print("[OK] Fallback: yolo11n.pt loaded on CPU")

# Initialize supervision annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

# Get token once for all channels
print(f"\nConnecting to NVR at {NVR_IP}...")
cam = Camera(NVR_IP, USERNAME, PASSWORD)
print(f"[OK] Connected! Token: {cam.token}")

# Open all 4 camera streams
cameras = {}
camera_locks = {}  # Thread locks for each camera

for channel in range(4):
    stream_url = f"http://{NVR_IP}/flv?port=1935&app=bcs&stream=channel{channel}_ext.bcs&token={cam.token}"
    print(f"\nOpening Channel {channel}...")
    print(f"  URL: {stream_url[:60]}...")

    cap = cv2.VideoCapture(stream_url)
    if cap.isOpened():
        print(f"  [OK] Channel {channel} opened successfully!")
        cameras[channel] = cap
        camera_locks[channel] = threading.Lock()  # Create lock for this camera
    else:
        print(f"  [FAIL] Channel {channel} failed to open")

print(f"\n[OK] {len(cameras)} cameras ready")


def generate_frames(channel):
    """Read frames from specific camera and yield as MJPEG with intelligent frame skipping"""
    cap = cameras.get(channel)
    lock = camera_locks.get(channel)

    if not cap or not lock:
        return

    frame_count = 0
    fps_start_time = time.time()

    # Frame synchronization variables
    last_yield_time = time.time()
    target_frame_time = 1.0 / 25.0  # Target 25 FPS output (0.04s per frame)
    detection_times = []  # Rolling window of recent detection times
    frames_skipped = 0
    frames_processed = 0

    while True:
        frame_start = time.time()

        # Use lock to ensure only one thread reads from this camera at a time
        with lock:
            success, frame = cap.read()

        if not success:
            print(f"Failed to read frame from channel {channel}")
            break

        frames_processed += 1

        # Apply detection if enabled
        global DETECTION_ENABLED
        detection_time = 0
        should_skip = False

        if DETECTION_ENABLED:
            # Calculate average detection time from recent measurements
            avg_detection_time = sum(detection_times[-10:]) / len(detection_times[-10:]) if detection_times else 0

            # Intelligent frame skipping logic:
            # Skip frames if we're falling behind (detection is slower than target frame time)
            # But only if we have enough data to make a decision (at least 5 frames processed)
            if len(detection_times) >= 5:
                # If detection takes longer than our target frame time, we need to skip frames
                if avg_detection_time > target_frame_time * 1.2:  # 20% tolerance
                    # Calculate how many frames behind we are
                    time_since_last_yield = time.time() - last_yield_time

                    # Skip this frame if we're not yet due for the next output
                    if time_since_last_yield < target_frame_time:
                        # Consume extra frames from buffer to stay synchronized
                        with lock:
                            for _ in range(min(3, int(cap.get(cv2.CAP_PROP_BUFFERSIZE) or 1))):
                                cap.grab()  # Discard buffered frames
                        frames_skipped += 1
                        should_skip = True

            if not should_skip:
                try:
                    det_start = time.time()

                    # Run YOLOv11 detection
                    results = model(frame, verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(results)

                    # Filter by confidence (>0.5)
                    detections = detections[detections.confidence > 0.5]

                    # Track metrics
                    with metrics_lock:
                        current_time = time.time()
                        if performance_metrics['start_time'] is None:
                            performance_metrics['start_time'] = current_time

                        # Record detections
                        for class_id, confidence in zip(detections.class_id, detections.confidence):
                            performance_metrics['detections'].append({
                                'timestamp': current_time,
                                'channel': channel,
                                'class': results.names[class_id],
                                'confidence': float(confidence)
                            })
                            performance_metrics['confidence'].append(float(confidence))
                            performance_metrics['total_detections'] += 1

                        # Keep only last 1000 detections
                        if len(performance_metrics['detections']) > 1000:
                            performance_metrics['detections'] = performance_metrics['detections'][-1000:]
                        if len(performance_metrics['confidence']) > 1000:
                            performance_metrics['confidence'] = performance_metrics['confidence'][-1000:]

                    # Annotate frame
                    frame = box_annotator.annotate(scene=frame, detections=detections)

                    # Create labels
                    labels = [
                        f"{results.names[class_id]} {confidence:.2f}"
                        for class_id, confidence in zip(detections.class_id, detections.confidence)
                    ]
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

                    detection_time = time.time() - det_start

                    # Update rolling window of detection times (keep last 20)
                    detection_times.append(detection_time)
                    if len(detection_times) > 20:
                        detection_times = detection_times[-20:]

                except Exception as e:
                    # If detection fails, just show the original frame
                    pass

        # Skip yielding this frame if we determined it should be skipped
        if should_skip:
            continue

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update FPS every 30 frames
            elapsed = time.time() - fps_start_time
            fps = 30 / elapsed if elapsed > 0 else 0

            with metrics_lock:
                performance_metrics['fps'][channel].append({
                    'timestamp': time.time(),
                    'fps': fps
                })
                # Keep only last 100 FPS measurements
                if len(performance_metrics['fps'][channel]) > 100:
                    performance_metrics['fps'][channel] = performance_metrics['fps'][channel][-100:]

                # Track processing time
                if detection_time > 0:
                    performance_metrics['processing_time'][channel].append({
                        'timestamp': time.time(),
                        'time': detection_time * 1000  # Convert to ms
                    })
                    if len(performance_metrics['processing_time'][channel]) > 100:
                        performance_metrics['processing_time'][channel] = performance_metrics['processing_time'][channel][-100:]

            # Track frame skip metrics
            with metrics_lock:
                performance_metrics['frames_skipped'][channel] += frames_skipped
                performance_metrics['frames_processed'][channel] += frames_processed

            # Log frame skip statistics if frames were skipped
            if frames_skipped > 0:
                skip_ratio = frames_skipped / frames_processed * 100
                print(f"[Channel {channel}] Skipped {frames_skipped}/{frames_processed} frames ({skip_ratio:.1f}%) for sync")

            frames_skipped = 0
            frames_processed = 0
            fps_start_time = time.time()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        # Update last yield time for synchronization
        last_yield_time = time.time()

        # Yield in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """4-camera grid view"""
    camera_count = len(cameras)

    # Build camera divs
    camera_divs = ""
    for i in range(4):
        if i in cameras:
            camera_divs += f'''
            <div class="camera">
                <h2>Camera {i+1} (Channel {i})</h2>
                <img src="/video_feed/{i}">
            </div>
            '''
        else:
            camera_divs += f'''
            <div class="camera offline">
                <h2>Camera {i+1} (Channel {i})</h2>
                <p style="color: #f00; text-align: center; padding: 50px;">OFFLINE</p>
            </div>
            '''

    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>4-Channel Camera Grid with Detection</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                background: #000;
                color: #0f0;
                font-family: monospace;
                padding: 20px;
            }}
            h1 {{
                color: #0f0;
                text-align: center;
                margin-bottom: 10px;
                text-shadow: 0 0 10px #0f0;
            }}
            .info {{
                text-align: center;
                margin-bottom: 20px;
                color: #0ff;
            }}
            .controls {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .toggle-btn {{
                background: #0f0;
                color: #000;
                border: 2px solid #0f0;
                padding: 12px 30px;
                font-size: 16px;
                font-family: monospace;
                font-weight: bold;
                cursor: pointer;
                border-radius: 5px;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
                transition: all 0.3s;
            }}
            .toggle-btn:hover {{
                background: #000;
                color: #0f0;
                box-shadow: 0 0 30px rgba(0, 255, 0, 0.8);
            }}
            .toggle-btn.active {{
                background: #f00;
                border-color: #f00;
                color: #fff;
                box-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
            }}
            .status {{
                display: inline-block;
                margin-left: 20px;
                padding: 8px 20px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }}
            .status.on {{
                background: #0f0;
                color: #000;
                box-shadow: 0 0 15px #0f0;
            }}
            .status.off {{
                background: #333;
                color: #666;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
                max-width: 1800px;
                margin: 0 auto;
            }}
            .camera {{
                background: #111;
                border: 2px solid #0f0;
                border-radius: 5px;
                padding: 10px;
                box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
            }}
            .camera h2 {{
                color: #0ff;
                text-align: center;
                margin-bottom: 10px;
                font-size: 1.2em;
            }}
            .camera img {{
                width: 100%;
                height: auto;
                display: block;
                border: 1px solid #0f0;
            }}
            .camera.offline {{
                border-color: #f00;
                box-shadow: 0 0 20px rgba(255, 0, 0, 0.3);
            }}
            .camera.offline h2 {{
                color: #f00;
            }}
            a {{
                color: #0ff;
                text-decoration: none;
                padding: 10px 20px;
                border: 2px solid #0ff;
                border-radius: 5px;
                display: inline-block;
                transition: all 0.3s;
            }}
            a:hover {{
                background: #0ff;
                color: #000;
            }}
        </style>
    </head>
    <body>
        <h1>📹 4-CHANNEL NVR CAMERA GRID 📹</h1>
        <div class="info">
            <p>NVR: {NVR_IP} | Model: {YOLO_MODEL} ({YOLO_DEVICE.upper()}) | Cameras: {camera_count}/4</p>
        </div>
        
        <div class="controls">
            <button class="toggle-btn" id="toggleBtn" onclick="toggleDetection()">
                🎯 TOGGLE DETECTION
            </button>
            <span class="status off" id="status">DETECTION: OFF</span>
            <a href="/analytics" style="margin-left: 20px;">📊 ANALYTICS</a>
        </div>
        
        <div class="grid">
            {camera_divs}
        </div>
        
        <script>
            let detectionEnabled = false;
            
            async function toggleDetection() {{
                try {{
                    const response = await fetch('/toggle_detection', {{
                        method: 'POST'
                    }});
                    const data = await response.json();
                    detectionEnabled = data.enabled;
                    
                    // Update UI
                    const btn = document.getElementById('toggleBtn');
                    const status = document.getElementById('status');
                    
                    if (detectionEnabled) {{
                        btn.classList.add('active');
                        status.classList.remove('off');
                        status.classList.add('on');
                        status.textContent = 'DETECTION: ON (YOLOv11)';
                    }} else {{
                        btn.classList.remove('active');
                        status.classList.remove('on');
                        status.classList.add('off');
                        status.textContent = 'DETECTION: OFF';
                    }}
                    
                    console.log('Detection toggled:', detectionEnabled);
                }} catch (error) {{
                    console.error('Error toggling detection:', error);
                }}
            }}
        </script>
    </body>
    </html>
    '''
    return html


@app.route('/video_feed/<int:channel>')
def video_feed(channel):
    """Video streaming route for specific channel"""
    if channel not in cameras:
        return "Camera not available", 404

    return Response(generate_frames(channel),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle YOLOv11 detection on/off"""
    global DETECTION_ENABLED
    with detection_lock:
        DETECTION_ENABLED = not DETECTION_ENABLED

    status = "enabled" if DETECTION_ENABLED else "disabled"
    print(f"\n[INFO] Detection {status}")

    return jsonify({
        'enabled': DETECTION_ENABLED,
        'status': status
    })


@app.route('/analytics')
def analytics():
    """Analytics dashboard with D3.js graphs"""
    with metrics_lock:
        # Calculate statistics
        total_detections = performance_metrics['total_detections']

        # Average confidence
        avg_confidence = sum(performance_metrics['confidence']) / len(performance_metrics['confidence']) if performance_metrics['confidence'] else 0

        # Average FPS per camera
        avg_fps = {}
        for ch in range(4):
            if performance_metrics['fps'][ch]:
                avg_fps[ch] = sum(f['fps'] for f in performance_metrics['fps'][ch]) / len(performance_metrics['fps'][ch])
            else:
                avg_fps[ch] = 0

        # Uptime
        uptime = time.time() - performance_metrics['start_time'] if performance_metrics['start_time'] else 0
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"

        # Detection rate (per minute)
        detection_rate = (total_detections / uptime * 60) if uptime > 0 else 0

    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Dashboard</title>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #000; color: #0f0; font-family: monospace; padding: 20px; }}
            h1 {{ color: #0f0; text-align: center; margin-bottom: 20px; text-shadow: 0 0 10px #0f0; }}
            .nav {{ text-align: center; margin-bottom: 20px; }}
            .nav a {{ color: #0ff; text-decoration: none; padding: 10px 20px; border: 2px solid #0ff; 
                     border-radius: 5px; margin: 0 10px; display: inline-block; transition: all 0.3s; }}
            .nav a:hover {{ background: #0ff; color: #000; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                          gap: 15px; margin-bottom: 30px; max-width: 1800px; margin-left: auto; margin-right: auto; }}
            .stat-card {{ background: #111; border: 2px solid #0f0; border-radius: 5px; padding: 20px;
                         text-align: center; box-shadow: 0 0 20px rgba(0, 255, 0, 0.3); }}
            .stat-value {{ font-size: 2.5em; color: #0ff; font-weight: bold; text-shadow: 0 0 10px #0ff; }}
            .stat-label {{ color: #0f0; margin-top: 10px; font-size: 0.9em; }}
            .charts-container {{ max-width: 1800px; margin: 0 auto; }}
            .chart {{ background: #111; border: 2px solid #0f0; border-radius: 5px; padding: 20px;
                     margin-bottom: 30px; box-shadow: 0 0 20px rgba(0, 255, 0, 0.3); }}
            .chart h2 {{ color: #0ff; margin-bottom: 15px; text-align: center; }}
            svg {{ display: block; margin: 0 auto; }}
            .line {{ fill: none; stroke-width: 2; }}
            .axis path, .axis line {{ stroke: #0f0; }}
            .axis text {{ fill: #0f0; font-family: monospace; }}
            .grid line {{ stroke: #333; stroke-opacity: 0.7; }}
            .bar {{ transition: opacity 0.3s; }}
            .bar:hover {{ opacity: 0.8; }}
        </style>
    </head>
    <body>
        <h1>📊 ANALYTICS DASHBOARD 📊</h1>
        
        <div class="nav">
            <a href="/">🎥 Camera View</a>
            <a href="/analytics">📊 Analytics</a>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="totalDetections">{total_detections}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgConfidence">{avg_confidence:.2f}</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="detectionRate">{detection_rate:.1f}</div>
                <div class="stat-label">Detections/Min</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="uptime">{uptime_str}</div>
                <div class="stat-label">Uptime</div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart">
                <h2>FPS Per Camera (Real-Time)</h2>
                <div id="fpsChart"></div>
            </div>
            
            <div class="chart">
                <h2>Detection Processing Time (ms)</h2>
                <div id="processingChart"></div>
            </div>
            
            <div class="chart">
                <h2>Detection Confidence Distribution</h2>
                <div id="confidenceChart"></div>
            </div>
            
            <div class="chart">
                <h2>Detections Over Time</h2>
                <div id="timelineChart"></div>
            </div>
        </div>
        
        <script>
            const width = 800;
            const height = 300;
            const margin = {{top: 20, right: 30, bottom: 50, left: 60}};
            
            async function fetchMetrics() {{
                try {{
                    const response = await fetch('/api/metrics');
                    const data = await response.json();
                    
                    document.getElementById('totalDetections').textContent = data.total_detections;
                    document.getElementById('avgConfidence').textContent = data.avg_confidence.toFixed(2);
                    document.getElementById('detectionRate').textContent = data.detection_rate.toFixed(1);
                    document.getElementById('uptime').textContent = data.uptime;
                    
                    drawFPSChart(data.fps);
                    drawProcessingChart(data.processing_time);
                    drawConfidenceChart(data.confidence);
                    drawTimelineChart(data.detections);
                }} catch (error) {{
                    console.error('Error fetching metrics:', error);
                }}
            }}
            
            function drawFPSChart(fpsData) {{
                d3.select('#fpsChart').selectAll('*').remove();
                const svg = d3.select('#fpsChart').append('svg').attr('width', width).attr('height', height);
                const colors = ['#00ff00', '#00ffff', '#ff00ff', '#ffff00'];
                
                Object.keys(fpsData).forEach((channel, i) => {{
                    if (fpsData[channel].length === 0) return;
                    const data = fpsData[channel];
                    const x = d3.scaleLinear().domain([0, data.length - 1]).range([margin.left, width - margin.right]);
                    const y = d3.scaleLinear().domain([0, 30]).range([height - margin.bottom, margin.top]);
                    const line = d3.line().x((d, i) => x(i)).y(d => y(d.fps));
                    svg.append('path').datum(data).attr('class', 'line').attr('d', line).attr('stroke', colors[parseInt(channel)]);
                }});
                
                const x = d3.scaleLinear().domain([0, 100]).range([margin.left, width - margin.right]);
                const y = d3.scaleLinear().domain([0, 30]).range([height - margin.bottom, margin.top]);
                svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(5));
                svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`).call(d3.axisLeft(y));
            }}
            
            function drawProcessingChart(processingData) {{
                d3.select('#processingChart').selectAll('*').remove();
                const svg = d3.select('#processingChart').append('svg').attr('width', width).attr('height', height);
                const colors = ['#00ff00', '#00ffff', '#ff00ff', '#ffff00'];
                
                const maxTime = d3.max(Object.values(processingData).flat(), d => d.time) || 100;
                Object.keys(processingData).forEach((channel, i) => {{
                    if (processingData[channel].length === 0) return;
                    const data = processingData[channel];
                    const x = d3.scaleLinear().domain([0, data.length - 1]).range([margin.left, width - margin.right]);
                    const y = d3.scaleLinear().domain([0, maxTime]).range([height - margin.bottom, margin.top]);
                    const line = d3.line().x((d, i) => x(i)).y(d => y(d.time));
                    svg.append('path').datum(data).attr('class', 'line').attr('d', line).attr('stroke', colors[parseInt(channel)]);
                }});
                
                const x = d3.scaleLinear().domain([0, 100]).range([margin.left, width - margin.right]);
                const y = d3.scaleLinear().domain([0, maxTime]).range([height - margin.bottom, margin.top]);
                svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(5));
                svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`).call(d3.axisLeft(y));
            }}
            
            function drawConfidenceChart(confidenceData) {{
                d3.select('#confidenceChart').selectAll('*').remove();
                if (confidenceData.length === 0) return;
                
                const svg = d3.select('#confidenceChart').append('svg').attr('width', width).attr('height', height);
                const bins = d3.bin().domain([0, 1]).thresholds(20)(confidenceData);
                const x = d3.scaleLinear().domain([0, 1]).range([margin.left, width - margin.right]);
                const y = d3.scaleLinear().domain([0, d3.max(bins, d => d.length)]).range([height - margin.bottom, margin.top]);
                
                svg.selectAll('rect').data(bins).join('rect').attr('class', 'bar')
                    .attr('x', d => x(d.x0) + 1).attr('width', d => Math.max(0, x(d.x1) - x(d.x0) - 2))
                    .attr('y', d => y(d.length)).attr('height', d => y(0) - y(d.length)).attr('fill', '#00ff00');
                
                svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(10));
                svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`).call(d3.axisLeft(y));
            }}
            
            function drawTimelineChart(detections) {{
                d3.select('#timelineChart').selectAll('*').remove();
                if (detections.length === 0) return;
                
                const svg = d3.select('#timelineChart').append('svg').attr('width', width).attr('height', height);
                const now = Date.now() / 1000;
                const timeRange = 300;
                const bucketSize = 10;
                
                const buckets = {{}};
                detections.forEach(d => {{
                    const bucket = Math.floor((now - d.timestamp) / bucketSize);
                    if (bucket >= 0 && bucket < timeRange / bucketSize) {{
                        if (!buckets[bucket]) buckets[bucket] = 0;
                        buckets[bucket]++;
                    }}
                }});
                
                const data = Object.keys(buckets).map(k => ({{time: parseInt(k), count: buckets[k]}})).sort((a, b) => b.time - a.time);
                const x = d3.scaleLinear().domain([timeRange / bucketSize, 0]).range([margin.left, width - margin.right]);
                const y = d3.scaleLinear().domain([0, d3.max(data, d => d.count) || 1]).range([height - margin.bottom, margin.top]);
                
                svg.selectAll('rect').data(data).join('rect').attr('class', 'bar')
                    .attr('x', d => x(d.time + 0.5) - 5).attr('width', 8)
                    .attr('y', d => y(d.count)).attr('height', d => y(0) - y(d.count)).attr('fill', '#00ffff');
                
                svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`)
                    .call(d3.axisBottom(x).ticks(6).tickFormat(d => `${{Math.round(d * bucketSize)}}s ago`));
                svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`).call(d3.axisLeft(y));
            }}
            
            fetchMetrics();
            setInterval(fetchMetrics, 2000);
        </script>
    </body>
    </html>
    '''
    return html


@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics data"""
    with metrics_lock:
        total_detections = performance_metrics['total_detections']
        avg_confidence = sum(performance_metrics['confidence']) / len(performance_metrics['confidence']) if performance_metrics['confidence'] else 0
        uptime = time.time() - performance_metrics['start_time'] if performance_metrics['start_time'] else 0
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
        detection_rate = (total_detections / uptime * 60) if uptime > 0 else 0

        # Calculate frame skip statistics
        skip_stats = {}
        for ch in range(4):
            total_frames = performance_metrics['frames_processed'][ch]
            skipped_frames = performance_metrics['frames_skipped'][ch]
            skip_ratio = (skipped_frames / total_frames * 100) if total_frames > 0 else 0
            skip_stats[ch] = {
                'skipped': skipped_frames,
                'total': total_frames,
                'ratio': round(skip_ratio, 2)
            }

        return jsonify({
            'total_detections': total_detections,
            'avg_confidence': avg_confidence,
            'detection_rate': detection_rate,
            'uptime': uptime_str,
            'fps': performance_metrics['fps'],
            'processing_time': performance_metrics['processing_time'],
            'confidence': performance_metrics['confidence'][-200:],
            'detections': performance_metrics['detections'][-200:],
            'frame_skip_stats': skip_stats
        })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting 4-camera grid viewer with YOLOv11 detection...")
    print(f"Configuration from .env:")
    print(f"  NVR IP: {NVR_IP}")
    print(f"  YOLO Model: {YOLO_MODEL}")
    print(f"  Device: {YOLO_DEVICE.upper()}")
    print("Open browser: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

