"""
4-Channel Camera Viewer with BATCHED YOLOv11 Detection
Processes all 4 camera frames in a single batch for better GPU utilization
Includes comprehensive performance metrics and comparison data
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
import numpy as np
import torch
from collections import deque

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
BATCHED_MODE = True  # Toggle for batched vs sequential processing
AUTO_FPS = True  # Automatically detect FPS from streams
TARGET_FPS = 12  # Manual FPS target (used when AUTO_FPS is False)
TRACKING_ENABLED = False  # Global toggle for object tracking with re-ID
detection_lock = threading.Lock()
config_lock = threading.Lock()

# Performance metrics tracking with batch-specific metrics
performance_metrics = {
    'fps': {i: [] for i in range(4)},  # FPS per camera
    'detections': [],  # Detection events with timestamp
    'confidence': [],  # Confidence scores
    'processing_time': {i: [] for i in range(4)},  # Detection processing time per camera
    'batch_processing_time': [],  # Time to process entire batch
    'batch_size': [],  # Actual batch size processed
    'frames_skipped': {i: 0 for i in range(4)},  # Total frames skipped per camera
    'frames_processed': {i: 0 for i in range(4)},  # Total frames processed per camera
    'total_detections': 0,
    'start_time': None,
    # Batch-specific metrics
    'gpu_utilization': [],  # GPU utilization percentage (if available)
    'memory_usage': [],  # GPU memory usage in MB
    'batch_efficiency': [],  # Ratio of batch processing time to sequential processing time estimate
    'preprocessing_time': [],  # Time spent preparing batch
    'postprocessing_time': [],  # Time spent splitting results
    'inference_time': [],  # Pure model inference time
}
metrics_lock = threading.Lock()

# Initialize YOLO model with configuration from environment
YOLO_MODEL = os.getenv('YOLO_MODEL', 'yolo11n.pt')
YOLO_DEVICE = os.getenv('YOLO_DEVICE', 'cpu')
BATCH_SIZE = 4  # Number of cameras we process in batch

print(f"\n[INFO] Loading {YOLO_MODEL} on {YOLO_DEVICE.upper()}...")
print(f"[INFO] Batch processing mode enabled for {BATCH_SIZE} cameras")

try:
    model = YOLO(YOLO_MODEL)
    if YOLO_DEVICE.lower() == 'cuda' or YOLO_DEVICE.startswith('0'):
        model.to('cuda')

        # Check if a TensorRT engine exists with proper batch size
        engine_path = f"{YOLO_MODEL.split('.')[0]}_batch{BATCH_SIZE}.engine"

        if os.path.exists(engine_path):
            print(f"[INFO] Found existing TensorRT engine: {engine_path}")
            try:
                model = YOLO(engine_path)
                print(f"[OK] Loaded TensorRT engine with batch size {BATCH_SIZE}")
                USE_GPU = True
            except Exception as e:
                print(f"[WARN] Failed to load existing engine: {e}")
                print(f"[INFO] Using PyTorch model on GPU instead")
                model = YOLO(YOLO_MODEL)
                model.to('cuda')
                USE_GPU = True
        else:
            model.export(format='engine', batch=4, dynamic=True)
            print(f"[INFO] No TensorRT engine found. Using PyTorch model on GPU")
            print(f"[INFO] For better performance, export with: model.export(format='engine', batch={BATCH_SIZE}, dynamic=True)")
            USE_GPU = True
    else:
        print(f"[OK] {YOLO_MODEL} loaded on CPU")
        USE_GPU = False
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    # Fallback to nano model on CPU
    model = YOLO("yolo11n.pt")
    print("[OK] Fallback: yolo11n.pt loaded on CPU")
    USE_GPU = False

# Initialize supervision annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

# Initialize ByteTrack trackers for each camera (separate tracker per camera for independent tracking)
print("\n[INFO] Initializing ByteTrack trackers with re-identification...")
trackers = {}
for channel in range(4):
    trackers[channel] = sv.ByteTrack(
        track_activation_threshold=0.5,  # Minimum confidence to start tracking
        lost_track_buffer=30,  # Frames to keep lost tracks in memory
        minimum_matching_threshold=0.8,  # IoU threshold for matching
        frame_rate=30,  # Assumed frame rate
        minimum_consecutive_frames=3  # Minimum frames to confirm a track
    )
print("[OK] ByteTrack trackers initialized for all channels")

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
        camera_locks[channel] = threading.Lock()
    else:
        print(f"  [FAIL] Channel {channel} failed to open")

print(f"\n[OK] {len(cameras)} cameras ready")

# Shared frame buffers for batch processing
latest_frames = {i: None for i in range(4)}
frames_lock = threading.Lock()

# Annotated frames ready for streaming
annotated_frames = {i: None for i in range(4)}
annotated_lock = threading.Lock()

# Frame queues for synchronization
frame_queues = {i: deque(maxlen=10) for i in range(4)}


def frame_capture_thread(channel):
    """Continuously capture frames from a specific camera and store in buffer"""
    cap = cameras.get(channel)
    lock = camera_locks.get(channel)

    if not cap or not lock:
        return

    frame_count = 0
    fps_start_time = time.time()

    while True:
        with lock:
            success, frame = cap.read()

        if not success:
            print(f"Failed to read frame from channel {channel}")
            time.sleep(0.1)
            continue

        # Store frame in buffer
        with frames_lock:
            latest_frames[channel] = frame.copy()
            frame_queues[channel].append((frame.copy(), time.time()))

        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start_time
            fps = 30 / elapsed if elapsed > 0 else 0

            with metrics_lock:
                performance_metrics['fps'][channel].append({
                    'timestamp': time.time(),
                    'fps': fps
                })
                if len(performance_metrics['fps'][channel]) > 100:
                    performance_metrics['fps'][channel] = performance_metrics['fps'][channel][-100:]

            fps_start_time = time.time()

        time.sleep(0.01)  # Small delay to prevent CPU overload


def batch_detection_thread():
    """Process all 4 camera frames in a single batch or sequentially based on BATCHED_MODE"""
    global DETECTION_ENABLED, BATCHED_MODE, AUTO_FPS, TARGET_FPS

    last_process_time = time.time()
    batch_times = []
    detected_fps = None

    while True:
        if not DETECTION_ENABLED:
            time.sleep(0.1)
            continue

        # Determine target FPS
        with config_lock:
            if AUTO_FPS and detected_fps is None:
                # Auto-detect FPS from first camera stream
                cap = cameras.get(0)
                if cap:
                    detected_fps = cap.get(cv2.CAP_PROP_FPS)
                    if detected_fps <= 0 or detected_fps > 60:
                        detected_fps = 20  # Default fallback
                    print(f"[AUTO FPS] Detected {detected_fps:.1f} FPS from camera stream")

            current_target_fps = detected_fps if (AUTO_FPS and detected_fps) else TARGET_FPS
            current_batched_mode = BATCHED_MODE
            target_frame_time = 1.0 / current_target_fps

        # Wait for target frame time
        time_since_last = time.time() - last_process_time
        if time_since_last < target_frame_time:
            time.sleep(target_frame_time - time_since_last)

        batch_start = time.time()

        # Collect frames from all cameras
        preprocess_start = time.time()
        frames_batch = []
        channels_in_batch = []

        with frames_lock:
            for channel in sorted(cameras.keys()):
                if latest_frames[channel] is not None:
                    frames_batch.append(latest_frames[channel].copy())
                    channels_in_batch.append(channel)

        if not frames_batch:
            time.sleep(0.01)
            continue

        preprocess_time = (time.time() - preprocess_start) * 1000  # ms

        # Run inference (batched or sequential based on mode)
        try:
            inference_start = time.time()

            if current_batched_mode:
                # BATCHED MODE: Process all frames in a single forward pass
                results_batch = model(frames_batch, verbose=False)
            else:
                # SEQUENTIAL MODE: Process each frame separately
                results_batch = []
                for frame in frames_batch:
                    result = model(frame, verbose=False)
                    results_batch.append(result[0] if isinstance(result, list) else result)

            inference_time = (time.time() - inference_start) * 1000  # ms

            # Post-process results
            postprocess_start = time.time()

            for i, (result, channel) in enumerate(zip(results_batch, channels_in_batch)):
                frame = frames_batch[i].copy()

                # Convert to supervision detections
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > 0.5]

                # Apply tracking if enabled
                if TRACKING_ENABLED:
                    detections = trackers[channel].update_with_detections(detections)

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
                            'class': result.names[class_id],
                            'confidence': float(confidence)
                        })
                        performance_metrics['confidence'].append(float(confidence))
                        performance_metrics['total_detections'] += 1

                    # Keep only last 1000 detections
                    if len(performance_metrics['detections']) > 1000:
                        performance_metrics['detections'] = performance_metrics['detections'][-1000:]
                    if len(performance_metrics['confidence']) > 1000:
                        performance_metrics['confidence'] = performance_metrics['confidence'][-1000:]

                    performance_metrics['frames_processed'][channel] += 1

                # Annotate frame
                annotated = box_annotator.annotate(scene=frame, detections=detections)

                # Create labels with tracker IDs if tracking is enabled
                if TRACKING_ENABLED and detections.tracker_id is not None:
                    labels = [
                        f"#{tracker_id} {result.names[class_id]} {confidence:.2f}"
                        for class_id, confidence, tracker_id
                        in zip(detections.class_id, detections.confidence, detections.tracker_id)
                    ]
                else:
                    labels = [
                        f"{result.names[class_id]} {confidence:.2f}"
                        for class_id, confidence in zip(detections.class_id, detections.confidence)
                    ]
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

                # Add processing mode info overlay
                mode_text = "BATCHED" if current_batched_mode else "SEQUENTIAL"
                tracking_text = " | TRACKING" if TRACKING_ENABLED else ""
                overlay_text = f"{mode_text}{tracking_text} | Cams: {len(frames_batch)} | {inference_time:.1f}ms | {current_target_fps:.0f} FPS target"
                cv2.putText(
                    annotated,
                    overlay_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255) if current_batched_mode else (255, 165, 0),
                    2
                )

                # Store annotated frame
                with annotated_lock:
                    annotated_frames[channel] = annotated

            postprocess_time = (time.time() - postprocess_start) * 1000  # ms

            # Calculate batch metrics
            batch_total_time = (time.time() - batch_start) * 1000  # ms
            batch_times.append(batch_total_time)
            if len(batch_times) > 50:
                batch_times = batch_times[-50:]

            # Estimate sequential processing time (batch_size * avg_time_per_frame)
            estimated_sequential_time = len(frames_batch) * (inference_time / len(frames_batch))
            efficiency = estimated_sequential_time / batch_total_time if batch_total_time > 0 else 1.0

            # Track batch-specific metrics
            with metrics_lock:
                performance_metrics['batch_processing_time'].append({
                    'timestamp': time.time(),
                    'time': batch_total_time
                })
                performance_metrics['batch_size'].append({
                    'timestamp': time.time(),
                    'size': len(frames_batch)
                })
                performance_metrics['batch_efficiency'].append({
                    'timestamp': time.time(),
                    'efficiency': efficiency
                })
                performance_metrics['preprocessing_time'].append({
                    'timestamp': time.time(),
                    'time': preprocess_time
                })
                performance_metrics['postprocessing_time'].append({
                    'timestamp': time.time(),
                    'time': postprocess_time
                })
                performance_metrics['inference_time'].append({
                    'timestamp': time.time(),
                    'time': inference_time
                })

                # GPU metrics if available
                if USE_GPU and torch.cuda.is_available():
                    try:
                        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        performance_metrics['memory_usage'].append({
                            'timestamp': time.time(),
                            'memory': memory_allocated
                        })
                    except:
                        pass

                # Keep only last 100 entries for each metric
                for key in ['batch_processing_time', 'batch_size', 'batch_efficiency',
                           'preprocessing_time', 'postprocessing_time', 'inference_time', 'memory_usage']:
                    if len(performance_metrics[key]) > 100:
                        performance_metrics[key] = performance_metrics[key][-100:]

            # Log batch statistics periodically
            if len(batch_times) % 10 == 0:
                avg_batch_time = sum(batch_times) / len(batch_times)
                print(f"[BATCH] Avg: {avg_batch_time:.1f}ms | Inference: {inference_time:.1f}ms | "
                      f"Efficiency: {efficiency:.2f}x | Batch: {len(frames_batch)}")

        except Exception as e:
            print(f"[ERROR] Batch detection failed: {e}")
            time.sleep(0.1)

        last_process_time = time.time()


def generate_frames(channel):
    """Stream annotated frames for specific channel - MULTI-USER SAFE

    This generator creates an independent video stream for each connected user.
    Each user gets their own copy of frames to prevent conflicts.
    """
    if channel not in cameras:
        return

    try:
        while True:
            frame = None

            try:
                if DETECTION_ENABLED:
                    # Stream annotated frames from batch processor
                    with annotated_lock:
                        annotated_frame = annotated_frames.get(channel)
                        if annotated_frame is not None:
                            # CRITICAL: Copy frame for this user to prevent conflicts
                            frame = annotated_frame.copy()

                    if frame is None:
                        # No annotated frame yet, use raw frame
                        with frames_lock:
                            raw_frame = latest_frames.get(channel)
                            if raw_frame is not None:
                                # CRITICAL: Copy frame for this user
                                frame = raw_frame.copy()
                                cv2.putText(
                                    frame,
                                    "BATCH MODE | Waiting for detection...",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 255),
                                    2
                                )
                else:
                    # Stream raw frames
                    with frames_lock:
                        raw_frame = latest_frames.get(channel)
                        if raw_frame is not None:
                            # CRITICAL: Copy frame for this user
                            frame = raw_frame.copy()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Encode frame as JPEG (each user gets their own encoding)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_bytes = buffer.tobytes()

                # Yield in multipart format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                # Small delay to prevent overwhelming the client
                time.sleep(0.001)

            except GeneratorExit:
                # Client disconnected gracefully
                print(f"[VIDEO STREAM] Client disconnected from channel {channel}")
                break
            except Exception as e:
                # Log error but continue trying
                print(f"[VIDEO STREAM] Error on channel {channel}: {e}")
                time.sleep(0.1)
                continue

    except Exception as e:
        print(f"[VIDEO STREAM] Fatal error on channel {channel}: {e}")
    finally:
        print(f"[VIDEO STREAM] Stream closed for channel {channel}")


@app.route('/')
def index():
    """4-camera grid view with batch processing info"""
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
        <title>4-Channel Camera Grid - BATCHED DETECTION</title>
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
            .batch-info {{
                background: #1a1a1a;
                border: 2px solid #ff0;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 0 20px rgba(255, 255, 0, 0.3);
            }}
            .batch-info h3 {{
                color: #ff0;
                margin-bottom: 10px;
            }}
            .batch-info p {{
                color: #0ff;
                margin: 5px 0;
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
        <h1>📹 4-CHANNEL NVR - BATCHED YOLO DETECTION 📹</h1>
        <div class="info">
            <p>NVR: {NVR_IP} | Model: {YOLO_MODEL} ({YOLO_DEVICE.upper()}) | Cameras: {camera_count}/4</p>
        </div>
        
        <div class="batch-info">
            <h3>⚡ BATCH PROCESSING MODE ⚡</h3>
            <p>All 4 camera frames are processed in a single GPU batch for optimal performance</p>
            <p id="batchStats">Batch stats loading...</p>
        </div>
        
        <div class="controls">
            <button class="toggle-btn" id="toggleBtn" onclick="toggleDetection()">
                🎯 TOGGLE DETECTION
            </button>
            <span class="status off" id="status">DETECTION: OFF</span>
            <br><br>
            <button class="toggle-btn" id="trackingBtn" onclick="toggleTracking()" style="background: #f0f; border-color: #f0f; color: #fff;">
                🎯 TRACKING
            </button>
            <span class="status off" id="trackingStatus" style="margin-left: 10px;">TRACKING: OFF</span>
            <br><br>
            <button class="toggle-btn" id="batchedModeBtn" onclick="toggleBatchedMode()" style="background: #ff0; border-color: #ff0; color: #000;">
                ⚡ BATCHED MODE
            </button>
            <button class="toggle-btn" id="autoFpsBtn" onclick="toggleAutoFps()" style="background: #0ff; border-color: #0ff; color: #000;">
                🔄 AUTO FPS
            </button>
            <span style="margin-left: 20px; color: #0ff;">
                Target FPS: 
                <input type="number" id="fpsInput" min="1" max="60" value="12" style="width: 60px; background: #111; color: #0f0; border: 2px solid #0f0; padding: 5px; font-family: monospace; font-size: 14px;">
                <button class="toggle-btn" onclick="setTargetFps()" style="padding: 8px 15px; font-size: 14px;">SET</button>
            </span>
            <br><br>
            <a href="/analytics">📊 ANALYTICS</a>
            <a href="/comparison" style="margin-left: 10px;">📈 PERFORMANCE COMPARISON</a>
        </div>
        
        <div class="grid">
            {camera_divs}
        </div>
        
        <script>
            let detectionEnabled = false;
            let trackingEnabled = false;
            
            async function toggleTracking() {{
                try {{
                    const response = await fetch('/toggle_tracking', {{
                        method: 'POST'
                    }});
                    const data = await response.json();
                    trackingEnabled = data.tracking_enabled;
                    
                    // Update UI
                    const btn = document.getElementById('trackingBtn');
                    const status = document.getElementById('trackingStatus');
                    
                    if (trackingEnabled) {{
                        btn.classList.add('active');
                        btn.style.background = '#0f0';
                        btn.style.borderColor = '#0f0';
                        btn.style.color = '#000';
                        status.classList.remove('off');
                        status.classList.add('on');
                        status.textContent = 'TRACKING: ON (ByteTrack + Re-ID)';
                    }} else {{
                        btn.classList.remove('active');
                        btn.style.background = '#f0f';
                        btn.style.borderColor = '#f0f';
                        btn.style.color = '#fff';
                        status.classList.remove('on');
                        status.classList.add('off');
                        status.textContent = 'TRACKING: OFF';
                    }}
                    
                    console.log('Tracking toggled:', trackingEnabled);
                }} catch (error) {{
                    console.error('Error toggling tracking:', error);
                }}
            }}
            
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
                        status.textContent = 'DETECTION: ON (BATCH MODE)';
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
            
            async function toggleBatchedMode() {{
                try {{
                    const response = await fetch('/toggle_batched_mode', {{
                        method: 'POST'
                    }});
                    const data = await response.json();
                    
                    const btn = document.getElementById('batchedModeBtn');
                    if (data.batched_mode) {{
                        btn.textContent = '⚡ BATCHED MODE';
                        btn.style.background = '#ff0';
                        btn.style.borderColor = '#ff0';
                    }} else {{
                        btn.textContent = '🔁 SEQUENTIAL MODE';
                        btn.style.background = '#f90';
                        btn.style.borderColor = '#f90';
                    }}
                    
                    console.log('Processing mode:', data.mode);
                }} catch (error) {{
                    console.error('Error toggling batched mode:', error);
                }}
            }}
            
            async function toggleAutoFps() {{
                try {{
                    const response = await fetch('/toggle_auto_fps', {{
                        method: 'POST'
                    }});
                    const data = await response.json();
                    
                    const btn = document.getElementById('autoFpsBtn');
                    const fpsInput = document.getElementById('fpsInput');
                    
                    if (data.auto_fps) {{
                        btn.textContent = '🔄 AUTO FPS';
                        btn.style.background = '#0ff';
                        btn.style.borderColor = '#0ff';
                        fpsInput.disabled = true;
                        fpsInput.style.opacity = '0.5';
                    }} else {{
                        btn.textContent = '✋ MANUAL FPS';
                        btn.style.background = '#f0f';
                        btn.style.borderColor = '#f0f';
                        fpsInput.disabled = false;
                        fpsInput.style.opacity = '1';
                    }}
                    
                    console.log('Auto FPS:', data.status);
                }} catch (error) {{
                    console.error('Error toggling auto FPS:', error);
                }}
            }}
            
            async function setTargetFps() {{
                const fps = parseFloat(document.getElementById('fpsInput').value);
                if (isNaN(fps) || fps < 1 || fps > 60) {{
                    alert('Please enter a valid FPS between 1 and 60');
                    return;
                }}
                
                try {{
                    const response = await fetch('/set_target_fps', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ fps: fps }})
                    }});
                    const data = await response.json();
                    
                    if (data.success) {{
                        console.log('Target FPS set to:', data.target_fps);
                        alert(`Target FPS set to ${{data.target_fps}}`);
                    }}
                }} catch (error) {{
                    console.error('Error setting target FPS:', error);
                }}
            }}
            
            async function updateConfig() {{
                try {{
                    const response = await fetch('/get_config');
                    const config = await response.json();
                    
                    // Update button states
                    const batchedBtn = document.getElementById('batchedModeBtn');
                    if (config.batched_mode) {{
                        batchedBtn.textContent = '⚡ BATCHED MODE';
                        batchedBtn.style.background = '#ff0';
                        batchedBtn.style.borderColor = '#ff0';
                    }} else {{
                        batchedBtn.textContent = '🔁 SEQUENTIAL MODE';
                        batchedBtn.style.background = '#f90';
                        batchedBtn.style.borderColor = '#f90';
                    }}
                    
                    const autoFpsBtn = document.getElementById('autoFpsBtn');
                    const fpsInput = document.getElementById('fpsInput');
                    if (config.auto_fps) {{
                        autoFpsBtn.textContent = '🔄 AUTO FPS';
                        autoFpsBtn.style.background = '#0ff';
                        autoFpsBtn.style.borderColor = '#0ff';
                        fpsInput.disabled = true;
                        fpsInput.style.opacity = '0.5';
                    }} else {{
                        autoFpsBtn.textContent = '✋ MANUAL FPS';
                        autoFpsBtn.style.background = '#f0f';
                        autoFpsBtn.style.borderColor = '#f0f';
                        fpsInput.disabled = false;
                        fpsInput.style.opacity = '1';
                    }}
                    
                    fpsInput.value = config.target_fps;
                }} catch (error) {{
                    console.error('Error fetching config:', error);
                }}
            }}
            
            async function updateBatchStats() {{
                try {{
                    const response = await fetch('/api/batch_stats');
                    const data = await response.json();
                    
                    document.getElementById('batchStats').innerHTML = 
                        `Avg Batch Time: ${{data.avg_batch_time}}ms | ` +
                        `Inference: ${{data.avg_inference_time}}ms | ` +
                        `Efficiency: ${{data.avg_efficiency}}x | ` +
                        `GPU Memory: ${{data.gpu_memory}}MB`;
                }} catch (error) {{
                    console.error('Error fetching batch stats:', error);
                }}
            }}
            
            // Initialize UI on load
            updateConfig();
            setInterval(updateBatchStats, 2000);
            updateBatchStats();
        </script>
    </body>
    </html>
    '''
    return html


@app.route('/video_feed/<int:channel>')
def video_feed(channel):
    """Video streaming route for specific channel - MULTI-USER SAFE"""
    if channel not in cameras:
        return "Camera not available", 404

    # Log connection for debugging
    client_ip = request.remote_addr
    print(f"[VIDEO STREAM] Channel {channel} connected from {client_ip}")

    return Response(generate_frames(channel),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle YOLOv11 batch detection on/off"""
    global DETECTION_ENABLED
    with detection_lock:
        DETECTION_ENABLED = not DETECTION_ENABLED

    status = "enabled" if DETECTION_ENABLED else "disabled"
    print(f"\n[INFO] Batch detection {status}")

    return jsonify({
        'enabled': DETECTION_ENABLED,
        'status': status
    })


@app.route('/toggle_batched_mode', methods=['POST'])
def toggle_batched_mode():
    """Toggle between batched and sequential processing"""
    global BATCHED_MODE
    with config_lock:
        BATCHED_MODE = not BATCHED_MODE

    mode = "batched" if BATCHED_MODE else "sequential"
    print(f"\n[INFO] Processing mode set to {mode}")

    return jsonify({
        'batched_mode': BATCHED_MODE,
        'mode': mode
    })


@app.route('/toggle_tracking', methods=['POST'])
def toggle_tracking():
    """Toggle object tracking with re-identification"""
    global TRACKING_ENABLED
    with config_lock:
        TRACKING_ENABLED = not TRACKING_ENABLED

    # Reset all trackers when toggling
    if TRACKING_ENABLED:
        print("\n[INFO] Object tracking enabled - ByteTrack with re-ID active")
    else:
        print("\n[INFO] Object tracking disabled")
        # Reset trackers to clear old IDs
        for channel in trackers:
            trackers[channel].reset()

    return jsonify({
        'tracking_enabled': TRACKING_ENABLED,
        'status': 'enabled' if TRACKING_ENABLED else 'disabled'
    })


@app.route('/toggle_auto_fps', methods=['POST'])
def toggle_auto_fps():
    """Toggle automatic FPS detection"""
    global AUTO_FPS
    with config_lock:
        AUTO_FPS = not AUTO_FPS

    status = "enabled" if AUTO_FPS else "disabled"
    print(f"\n[INFO] Auto FPS {status}")

    return jsonify({
        'auto_fps': AUTO_FPS,
        'status': status
    })


@app.route('/set_target_fps', methods=['POST'])
def set_target_fps():
    """Set manual target FPS"""
    global TARGET_FPS
    data = request.get_json()
    fps = float(data.get('fps', 12))

    # Clamp FPS between 1 and 60
    fps = max(1, min(60, fps))

    with config_lock:
        TARGET_FPS = fps

    print(f"\n[INFO] Target FPS set to {fps}")

    return jsonify({
        'target_fps': TARGET_FPS,
        'success': True
    })


@app.route('/get_config', methods=['GET'])
def get_config():
    """Get current configuration"""
    with config_lock:
        return jsonify({
            'detection_enabled': DETECTION_ENABLED,
            'batched_mode': BATCHED_MODE,
            'auto_fps': AUTO_FPS,
            'target_fps': TARGET_FPS
        })


@app.route('/api/batch_stats')
def api_batch_stats():
    """Quick batch statistics for main page"""
    with metrics_lock:
        avg_batch_time = 0
        if performance_metrics['batch_processing_time']:
            recent = performance_metrics['batch_processing_time'][-10:]
            avg_batch_time = sum(x['time'] for x in recent) / len(recent)

        avg_inference_time = 0
        if performance_metrics['inference_time']:
            recent = performance_metrics['inference_time'][-10:]
            avg_inference_time = sum(x['time'] for x in recent) / len(recent)

        avg_efficiency = 0
        if performance_metrics['batch_efficiency']:
            recent = performance_metrics['batch_efficiency'][-10:]
            avg_efficiency = sum(x['efficiency'] for x in recent) / len(recent)

        gpu_memory = 0
        if performance_metrics['memory_usage']:
            gpu_memory = performance_metrics['memory_usage'][-1]['memory']

        return jsonify({
            'avg_batch_time': f"{avg_batch_time:.1f}",
            'avg_inference_time': f"{avg_inference_time:.1f}",
            'avg_efficiency': f"{avg_efficiency:.2f}",
            'gpu_memory': f"{gpu_memory:.0f}"
        })


@app.route('/analytics')
def analytics():
    """Enhanced analytics dashboard with batch-specific metrics"""
    with metrics_lock:
        total_detections = performance_metrics['total_detections']
        avg_confidence = sum(performance_metrics['confidence']) / len(performance_metrics['confidence']) if performance_metrics['confidence'] else 0
        uptime = time.time() - performance_metrics['start_time'] if performance_metrics['start_time'] else 0
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
        detection_rate = (total_detections / uptime * 60) if uptime > 0 else 0

        # Batch-specific stats
        avg_batch_time = 0
        if performance_metrics['batch_processing_time']:
            avg_batch_time = sum(x['time'] for x in performance_metrics['batch_processing_time']) / len(performance_metrics['batch_processing_time'])

        avg_efficiency = 0
        if performance_metrics['batch_efficiency']:
            avg_efficiency = sum(x['efficiency'] for x in performance_metrics['batch_efficiency']) / len(performance_metrics['batch_efficiency'])

    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Processing Analytics</title>
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
            .stat-card.batch {{ border-color: #ff0; box-shadow: 0 0 20px rgba(255, 255, 0, 0.3); }}
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
        </style>
    </head>
    <body>
        <h1>📊 BATCH PROCESSING ANALYTICS 📊</h1>
        
        <div class="nav">
            <a href="/">🎥 Camera View</a>
            <a href="/analytics">📊 Analytics</a>
            <a href="/comparison">📈 Performance Comparison</a>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="totalDetections">{total_detections}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card batch">
                <div class="stat-value" id="avgBatchTime">{avg_batch_time:.1f}ms</div>
                <div class="stat-label">Avg Batch Time</div>
            </div>
            <div class="stat-card batch">
                <div class="stat-value" id="avgEfficiency">{avg_efficiency:.2f}x</div>
                <div class="stat-label">Batch Efficiency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="uptime">{uptime_str}</div>
                <div class="stat-label">Uptime</div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart">
                <h2>Batch Processing Time Breakdown</h2>
                <div id="batchBreakdown"></div>
            </div>
            
            <div class="chart">
                <h2>Batch Efficiency Over Time</h2>
                <div id="efficiencyChart"></div>
            </div>
            
            <div class="chart">
                <h2>GPU Memory Usage</h2>
                <div id="memoryChart"></div>
            </div>
            
            <div class="chart">
                <h2>FPS Per Camera (with Batch Processing)</h2>
                <div id="fpsChart"></div>
            </div>
            
            <div class="chart">
                <h2>Inference Time vs Total Batch Time</h2>
                <div id="timeComparisonChart"></div>
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
                    document.getElementById('avgBatchTime').textContent = data.avg_batch_time.toFixed(1) + 'ms';
                    document.getElementById('avgEfficiency').textContent = data.avg_efficiency.toFixed(2) + 'x';
                    document.getElementById('uptime').textContent = data.uptime;
                    
                    drawBatchBreakdown(data);
                    drawEfficiencyChart(data.batch_efficiency);
                    drawMemoryChart(data.memory_usage);
                    drawFPSChart(data.fps);
                    drawTimeComparisonChart(data);
                }} catch (error) {{
                    console.error('Error fetching metrics:', error);
                }}
            }}
            
            function drawBatchBreakdown(data) {{
                d3.select('#batchBreakdown').selectAll('*').remove();
                const svg = d3.select('#batchBreakdown').append('svg').attr('width', width).attr('height', height);
                
                const breakdown = [
                    {{label: 'Preprocess', value: data.avg_preprocessing_time}},
                    {{label: 'Inference', value: data.avg_inference_time}},
                    {{label: 'Postprocess', value: data.avg_postprocessing_time}}
                ];
                
                const x = d3.scaleBand().domain(breakdown.map(d => d.label)).range([margin.left, width - margin.right]).padding(0.3);
                const y = d3.scaleLinear().domain([0, d3.max(breakdown, d => d.value)]).range([height - margin.bottom, margin.top]);
                
                svg.selectAll('rect').data(breakdown).join('rect')
                    .attr('x', d => x(d.label)).attr('width', x.bandwidth())
                    .attr('y', d => y(d.value)).attr('height', d => y(0) - y(d.value))
                    .attr('fill', (d, i) => ['#ff0', '#0ff', '#f0f'][i]);
                
                svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x));
                svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`).call(d3.axisLeft(y));
            }}
            
            function drawEfficiencyChart(efficiencyData) {{
                d3.select('#efficiencyChart').selectAll('*').remove();
                if (efficiencyData.length === 0) return;
                
                const svg = d3.select('#efficiencyChart').append('svg').attr('width', width).attr('height', height);
                const x = d3.scaleLinear().domain([0, efficiencyData.length - 1]).range([margin.left, width - margin.right]);
                const y = d3.scaleLinear().domain([0, d3.max(efficiencyData, d => d.efficiency)]).range([height - margin.bottom, margin.top]);
                const line = d3.line().x((d, i) => x(i)).y(d => y(d.efficiency));
                
                svg.append('path').datum(efficiencyData).attr('class', 'line').attr('d', line).attr('stroke', '#ff0');
                svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(5));
                svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`).call(d3.axisLeft(y));
            }}
            
            function drawMemoryChart(memoryData) {{
                d3.select('#memoryChart').selectAll('*').remove();
                if (memoryData.length === 0) return;
                
                const svg = d3.select('#memoryChart').append('svg').attr('width', width).attr('height', height);
                const x = d3.scaleLinear().domain([0, memoryData.length - 1]).range([margin.left, width - margin.right]);
                const y = d3.scaleLinear().domain([0, d3.max(memoryData, d => d.memory)]).range([height - margin.bottom, margin.top]);
                const line = d3.line().x((d, i) => x(i)).y(d => y(d.memory));
                
                svg.append('path').datum(memoryData).attr('class', 'line').attr('d', line).attr('stroke', '#0ff');
                svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(5));
                svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`).call(d3.axisLeft(y));
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
            
            function drawTimeComparisonChart(data) {{
                d3.select('#timeComparisonChart').selectAll('*').remove();
                if (data.inference_time.length === 0) return;
                
                const svg = d3.select('#timeComparisonChart').append('svg').attr('width', width).attr('height', height);
                const inferenceData = data.inference_time.slice(-50);
                const batchData = data.batch_processing_time.slice(-50);
                
                const x = d3.scaleLinear().domain([0, Math.max(inferenceData.length, batchData.length) - 1]).range([margin.left, width - margin.right]);
                const y = d3.scaleLinear().domain([0, d3.max([...batchData.map(d => d.time), ...inferenceData.map(d => d.time)])]).range([height - margin.bottom, margin.top]);
                
                const line = d3.line().x((d, i) => x(i)).y(d => y(d.time));
                svg.append('path').datum(inferenceData).attr('class', 'line').attr('d', line).attr('stroke', '#0ff');
                svg.append('path').datum(batchData).attr('class', 'line').attr('d', line).attr('stroke', '#f0f');
                
                svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`).call(d3.axisBottom(x).ticks(5));
                svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`).call(d3.axisLeft(y));
            }}
            
            fetchMetrics();
            setInterval(fetchMetrics, 2000);
        </script>
    </body>
    </html>
    '''
    return html


@app.route('/comparison')
def comparison():
    """Performance comparison and recommendations page"""
    with metrics_lock:
        # Calculate comprehensive statistics
        avg_batch_time = 0
        if performance_metrics['batch_processing_time']:
            avg_batch_time = sum(x['time'] for x in performance_metrics['batch_processing_time']) / len(performance_metrics['batch_processing_time'])

        avg_inference_time = 0
        if performance_metrics['inference_time']:
            avg_inference_time = sum(x['time'] for x in performance_metrics['inference_time']) / len(performance_metrics['inference_time'])

        avg_preprocessing_time = 0
        if performance_metrics['preprocessing_time']:
            avg_preprocessing_time = sum(x['time'] for x in performance_metrics['preprocessing_time']) / len(performance_metrics['preprocessing_time'])

        avg_postprocessing_time = 0
        if performance_metrics['postprocessing_time']:
            avg_postprocessing_time = sum(x['time'] for x in performance_metrics['postprocessing_time']) / len(performance_metrics['postprocessing_time'])

        avg_efficiency = 0
        if performance_metrics['batch_efficiency']:
            avg_efficiency = sum(x['efficiency'] for x in performance_metrics['batch_efficiency']) / len(performance_metrics['batch_efficiency'])

        # Estimate sequential processing time
        estimated_sequential_time = avg_inference_time * 4  # 4 cameras processed separately
        time_saved = estimated_sequential_time - avg_batch_time
        time_saved_percent = (time_saved / estimated_sequential_time * 100) if estimated_sequential_time > 0 else 0

        # Generate recommendations
        recommendations = []

        # Check if we have any data yet
        if avg_batch_time == 0 and avg_inference_time == 0:
            recommendations.append({
                'level': 'info',
                'title': 'Waiting for Data',
                'message': 'No batch processing data available yet. Enable detection and wait a few seconds for metrics to accumulate.',
                'actions': [
                    'Click "TOGGLE DETECTION" on the main page to start batch processing',
                    'Wait 5-10 seconds for metrics to be collected',
                    'Refresh this page to see performance data'
                ]
            })

        if avg_efficiency < 2.0 and avg_batch_time > 0:
            recommendations.append({
                'level': 'warning',
                'title': 'Low Batch Efficiency',
                'message': f'Batch efficiency is {avg_efficiency:.2f}x, which is lower than expected. This suggests overhead or GPU underutilization.',
                'actions': [
                    'Consider using a larger YOLO model (e.g., yolo11l or yolo11x) to increase GPU utilization',
                    'Check if GPU is properly configured and CUDA is available',
                    'Monitor GPU utilization to identify bottlenecks'
                ]
            })
        elif avg_efficiency >= 2.0 and avg_batch_time > 0:
            recommendations.append({
                'level': 'success',
                'title': 'Good Batch Efficiency',
                'message': f'Batch efficiency of {avg_efficiency:.2f}x indicates effective parallelization.',
                'actions': []
            })

        if avg_batch_time > 100:
            recommendations.append({
                'level': 'warning',
                'title': 'High Batch Processing Time',
                'message': f'Average batch time of {avg_batch_time:.1f}ms may cause frame drops.',
                'actions': [
                    'Consider using a smaller/faster model (yolo11n or yolo11s)',
                    'Reduce input resolution if acceptable',
                    'Enable TensorRT or ONNX optimization',
                    'Increase target_fps in batch_detection_thread to skip more frames'
                ]
            })

        if avg_preprocessing_time > avg_inference_time * 0.2 and avg_batch_time > 0:
            preprocessing_percent = (avg_preprocessing_time/avg_batch_time*100) if avg_batch_time > 0 else 0
            recommendations.append({
                'level': 'info',
                'title': 'Preprocessing Overhead',
                'message': f'Preprocessing takes {avg_preprocessing_time:.1f}ms ({preprocessing_percent:.1f}% of total time).',
                'actions': [
                    'Preprocessing is relatively low - this is expected',
                    'Frame copying is necessary for thread safety'
                ]
            })

        if USE_GPU:
            gpu_memory = 0
            if performance_metrics['memory_usage']:
                gpu_memory = performance_metrics['memory_usage'][-1]['memory']

            if gpu_memory > 0:
                recommendations.append({
                    'level': 'info',
                    'title': 'GPU Memory Usage',
                    'message': f'Currently using {gpu_memory:.0f}MB of GPU memory.',
                    'actions': [
                        'Monitor memory usage to ensure no OOM errors',
                        'If memory is limited, use a smaller model or reduce batch size'
                    ]
                })
        else:
            recommendations.append({
                'level': 'critical',
                'title': 'CPU Mode Detected',
                'message': 'Running on CPU significantly reduces batch processing benefits.',
                'actions': [
                    'Install CUDA and PyTorch with GPU support for massive speedup',
                    'Set YOLO_DEVICE=cuda in .env file',
                    'Expected 10-50x performance improvement with GPU'
                ]
            })

    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Performance Comparison & Recommendations</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #000; color: #0f0; font-family: monospace; padding: 20px; }}
            h1 {{ color: #0f0; text-align: center; margin-bottom: 20px; text-shadow: 0 0 10px #0f0; }}
            .nav {{ text-align: center; margin-bottom: 20px; }}
            .nav a {{ color: #0ff; text-decoration: none; padding: 10px 20px; border: 2px solid #0ff; 
                     border-radius: 5px; margin: 0 10px; display: inline-block; transition: all 0.3s; }}
            .nav a:hover {{ background: #0ff; color: #000; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ background: #111; border: 2px solid #0f0; border-radius: 5px; padding: 20px;
                       margin-bottom: 20px; box-shadow: 0 0 20px rgba(0, 255, 0, 0.3); }}
            .section h2 {{ color: #0ff; margin-bottom: 15px; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            .metrics-table th, .metrics-table td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
            .metrics-table th {{ color: #ff0; }}
            .metrics-table td {{ color: #0ff; }}
            .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .comparison-card {{ background: #1a1a1a; padding: 20px; border-radius: 5px; border: 2px solid #0f0; }}
            .comparison-card h3 {{ color: #ff0; margin-bottom: 10px; }}
            .comparison-card .value {{ font-size: 2em; color: #0ff; font-weight: bold; }}
            .recommendation {{ background: #1a1a1a; padding: 15px; margin-bottom: 15px; border-radius: 5px; border-left: 4px solid; }}
            .recommendation.success {{ border-color: #0f0; }}
            .recommendation.warning {{ border-color: #ff0; }}
            .recommendation.critical {{ border-color: #f00; }}
            .recommendation.info {{ border-color: #0ff; }}
            .recommendation h3 {{ margin-bottom: 10px; }}
            .recommendation.success h3 {{ color: #0f0; }}
            .recommendation.warning h3 {{ color: #ff0; }}
            .recommendation.critical h3 {{ color: #f00; }}
            .recommendation.info h3 {{ color: #0ff; }}
            .recommendation ul {{ margin-left: 20px; margin-top: 10px; }}
            .recommendation li {{ margin: 5px 0; color: #0f0; }}
            .auto-refresh {{ text-align: center; padding: 10px; background: #1a1a1a; 
                           border: 2px solid #0ff; border-radius: 5px; margin-bottom: 20px; }}
            .auto-refresh span {{ color: #0ff; font-weight: bold; }}
            .pulse {{ animation: pulse 2s infinite; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        </style>
    </head>
    <body>
        <h1>📈 PERFORMANCE COMPARISON & RECOMMENDATIONS 📈</h1>
        
        <div class="auto-refresh">
            <span class="pulse">🔄 AUTO-REFRESHING</span> | 
            Last Update: <span id="lastUpdate">Loading...</span> | 
            Next Update: <span id="countdown">2s</span>
        </div>
        
        <div class="nav">
            <a href="/">🎥 Camera View</a>
            <a href="/analytics">📊 Analytics</a>
            <a href="/comparison">📈 Performance Comparison</a>
        </div>
        
        <div class="container">
            <div class="section">
                <h2>Batch Processing Statistics</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Description</th>
                    </tr>
                    <tr>
                        <td>Average Batch Time</td>
                        <td id="avgBatchTime">{avg_batch_time:.2f}ms</td>
                        <td>Total time to process all 4 cameras in one batch</td>
                    </tr>
                    <tr>
                        <td>Inference Time</td>
                        <td id="avgInferenceTime">{avg_inference_time:.2f}ms</td>
                        <td>Pure model inference time for the batch</td>
                    </tr>
                    <tr>
                        <td>Preprocessing Time</td>
                        <td id="avgPreprocessingTime">{avg_preprocessing_time:.2f}ms</td>
                        <td>Time to collect and prepare frames</td>
                    </tr>
                    <tr>
                        <td>Postprocessing Time</td>
                        <td id="avgPostprocessingTime">{avg_postprocessing_time:.2f}ms</td>
                        <td>Time to annotate and distribute results</td>
                    </tr>
                    <tr>
                        <td>Batch Efficiency</td>
                        <td id="avgEfficiency">{avg_efficiency:.2f}x</td>
                        <td>Speedup vs sequential processing</td>
                    </tr>
                    <tr>
                        <td>Device</td>
                        <td id="device">{"GPU (CUDA)" if USE_GPU else "CPU"}</td>
                        <td>Hardware being used for inference</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Batch vs Sequential Comparison</h2>
                <div class="comparison">
                    <div class="comparison-card">
                        <h3>Estimated Sequential Time</h3>
                        <div class="value" id="estimatedSeqTime">{estimated_sequential_time:.1f}ms</div>
                        <p>Processing 4 cameras one at a time</p>
                    </div>
                    <div class="comparison-card">
                        <h3>Actual Batch Time</h3>
                        <div class="value" id="actualBatchTime">{avg_batch_time:.1f}ms</div>
                        <p>Processing all 4 cameras together</p>
                    </div>
                </div>
                <div class="comparison">
                    <div class="comparison-card">
                        <h3>Time Saved</h3>
                        <div class="value" id="timeSaved">{time_saved:.1f}ms</div>
                        <p id="timeSavedPercent">{time_saved_percent:.1f}% faster</p>
                    </div>
                    <div class="comparison-card">
                        <h3>Frames Per Second</h3>
                        <div class="value" id="maxFps">{(1000/avg_batch_time if avg_batch_time > 0 else 0):.1f} FPS</div>
                        <p>Maximum detection rate</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations & Optimization Tips</h2>
                {''.join(f"""
                <div class="recommendation {rec['level']}">
                    <h3>{rec['title']}</h3>
                    <p>{rec['message']}</p>
                    {'<ul>' + ''.join(f'<li>{action}</li>' for action in rec['actions']) + '</ul>' if rec['actions'] else ''}
                </div>
                """ for rec in recommendations)}
            </div>
            
            <div class="section">
                <h2>Understanding Batch Processing Benefits</h2>
                <p style="margin-bottom: 15px;">
                    Batch processing combines multiple frames into a single inference operation, which provides several benefits:
                </p>
                <ul style="margin-left: 20px; color: #0f0;">
                    <li><strong>Reduced Overhead:</strong> GPU dispatch and initialization happens once for all frames</li>
                    <li><strong>Better GPU Utilization:</strong> Parallel processing of multiple frames uses more GPU cores</li>
                    <li><strong>Memory Efficiency:</strong> Single memory transfer for all frames</li>
                    <li><strong>Throughput Optimization:</strong> Higher overall detection rate across all cameras</li>
                </ul>
                <p style="margin-top: 15px; color: #ff0;">
                    <strong>Note:</strong> Batch processing is most effective with GPU acceleration. On CPU, benefits are minimal.
                </p>
            </div>
            
            <div class="section">
                <h2>Further Optimization Ideas</h2>
                <ul style="margin-left: 20px; color: #0f0;">
                    <li><strong>TensorRT Optimization:</strong> Export model to TensorRT for 2-5x additional speedup on NVIDIA GPUs</li>
                    <li><strong>Mixed Precision (FP16):</strong> Use half-precision for faster inference with minimal accuracy loss</li>
                    <li><strong>Dynamic Batching:</strong> Adjust batch size based on available frames and processing time</li>
                    <li><strong>Asynchronous Processing:</strong> Use CUDA streams for overlapped computation</li>
                    <li><strong>Model Quantization:</strong> INT8 quantization can provide 2-4x speedup with proper calibration</li>
                    <li><strong>Resize Optimization:</strong> Pre-resize frames to model input size before batching</li>
                </ul>
            </div>
        </div>
        
        <script>
            let countdown = 2;
            let countdownInterval;
            
            function updateTimestamp() {{
                const now = new Date();
                const timeStr = now.toLocaleTimeString();
                document.getElementById('lastUpdate').textContent = timeStr;
            }}
            
            function startCountdown() {{
                countdown = 2;
                if (countdownInterval) {{
                    clearInterval(countdownInterval);
                }}
                countdownInterval = setInterval(() => {{
                    countdown -= 0.1;
                    if (countdown <= 0) {{
                        countdown = 2;
                    }}
                    document.getElementById('countdown').textContent = countdown.toFixed(1) + 's';
                }}, 100);
            }}
            
            async function updateMetrics() {{
                try {{
                    const response = await fetch('/api/comparison_metrics');
                    const data = await response.json();
                    
                    // Update metrics table
                    document.getElementById('avgBatchTime').textContent = data.avg_batch_time.toFixed(2) + 'ms';
                    document.getElementById('avgInferenceTime').textContent = data.avg_inference_time.toFixed(2) + 'ms';
                    document.getElementById('avgPreprocessingTime').textContent = data.avg_preprocessing_time.toFixed(2) + 'ms';
                    document.getElementById('avgPostprocessingTime').textContent = data.avg_postprocessing_time.toFixed(2) + 'ms';
                    document.getElementById('avgEfficiency').textContent = data.avg_efficiency.toFixed(2) + 'x';
                    
                    // Update comparison cards
                    document.getElementById('estimatedSeqTime').textContent = data.estimated_sequential_time.toFixed(1) + 'ms';
                    document.getElementById('actualBatchTime').textContent = data.avg_batch_time.toFixed(1) + 'ms';
                    document.getElementById('timeSaved').textContent = data.time_saved.toFixed(1) + 'ms';
                    document.getElementById('timeSavedPercent').textContent = data.time_saved_percent.toFixed(1) + '% faster';
                    document.getElementById('maxFps').textContent = data.max_fps.toFixed(1) + ' FPS';
                    
                    // Update timestamp
                    updateTimestamp();
                    
                    // Visual feedback - flash updated values
                    const elements = [
                        'avgBatchTime', 'avgInferenceTime', 'avgPreprocessingTime', 
                        'avgPostprocessingTime', 'avgEfficiency', 'estimatedSeqTime',
                        'actualBatchTime', 'timeSaved', 'timeSavedPercent', 'maxFps'
                    ];
                    elements.forEach(id => {{
                        const elem = document.getElementById(id);
                        elem.style.transition = 'color 0.3s';
                        elem.style.color = '#ff0';
                        setTimeout(() => {{
                            elem.style.color = '#0ff';
                        }}, 300);
                    }});
                    
                }} catch (error) {{
                    console.error('Error fetching metrics:', error);
                    document.getElementById('lastUpdate').textContent = 'Error - Retrying...';
                }}
            }}
            
            // Initial update
            updateTimestamp();
            updateMetrics();
            startCountdown();
            
            // Update every 2 seconds
            setInterval(() => {{
                updateMetrics();
            }}, 2000);
            
            console.log('Auto-refresh enabled - Updates every 2 seconds');
        </script>
    </body>
    </html>
    '''
    return html


@app.route('/api/comparison_metrics')
def api_comparison_metrics():
    """API endpoint for comparison page metrics - supports auto-refresh"""
    with metrics_lock:
        # Calculate comprehensive statistics
        avg_batch_time = 0
        if performance_metrics['batch_processing_time']:
            avg_batch_time = sum(x['time'] for x in performance_metrics['batch_processing_time']) / len(performance_metrics['batch_processing_time'])

        avg_inference_time = 0
        if performance_metrics['inference_time']:
            avg_inference_time = sum(x['time'] for x in performance_metrics['inference_time']) / len(performance_metrics['inference_time'])

        avg_preprocessing_time = 0
        if performance_metrics['preprocessing_time']:
            avg_preprocessing_time = sum(x['time'] for x in performance_metrics['preprocessing_time']) / len(performance_metrics['preprocessing_time'])

        avg_postprocessing_time = 0
        if performance_metrics['postprocessing_time']:
            avg_postprocessing_time = sum(x['time'] for x in performance_metrics['postprocessing_time']) / len(performance_metrics['postprocessing_time'])

        avg_efficiency = 0
        if performance_metrics['batch_efficiency']:
            avg_efficiency = sum(x['efficiency'] for x in performance_metrics['batch_efficiency']) / len(performance_metrics['batch_efficiency'])

        # Estimate sequential processing time
        estimated_sequential_time = avg_inference_time * 4
        time_saved = estimated_sequential_time - avg_batch_time
        time_saved_percent = (time_saved / estimated_sequential_time * 100) if estimated_sequential_time > 0 else 0

        # GPU memory if available
        gpu_memory = 0
        if performance_metrics['memory_usage']:
            gpu_memory = performance_metrics['memory_usage'][-1]['memory']

        return jsonify({
            'avg_batch_time': avg_batch_time,
            'avg_inference_time': avg_inference_time,
            'avg_preprocessing_time': avg_preprocessing_time,
            'avg_postprocessing_time': avg_postprocessing_time,
            'avg_efficiency': avg_efficiency,
            'estimated_sequential_time': estimated_sequential_time,
            'time_saved': time_saved,
            'time_saved_percent': time_saved_percent,
            'max_fps': (1000 / avg_batch_time) if avg_batch_time > 0 else 0,
            'gpu_memory': gpu_memory,
            'use_gpu': USE_GPU,
            'has_data': avg_batch_time > 0 or avg_inference_time > 0
        })


@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics data"""
    with metrics_lock:
        total_detections = performance_metrics['total_detections']
        avg_confidence = sum(performance_metrics['confidence']) / len(performance_metrics['confidence']) if performance_metrics['confidence'] else 0
        uptime = time.time() - performance_metrics['start_time'] if performance_metrics['start_time'] else 0
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
        detection_rate = (total_detections / uptime * 60) if uptime > 0 else 0

        # Batch-specific metrics
        avg_batch_time = 0
        if performance_metrics['batch_processing_time']:
            avg_batch_time = sum(x['time'] for x in performance_metrics['batch_processing_time']) / len(performance_metrics['batch_processing_time'])

        avg_efficiency = 0
        if performance_metrics['batch_efficiency']:
            avg_efficiency = sum(x['efficiency'] for x in performance_metrics['batch_efficiency']) / len(performance_metrics['batch_efficiency'])

        avg_preprocessing_time = 0
        if performance_metrics['preprocessing_time']:
            avg_preprocessing_time = sum(x['time'] for x in performance_metrics['preprocessing_time']) / len(performance_metrics['preprocessing_time'])

        avg_postprocessing_time = 0
        if performance_metrics['postprocessing_time']:
            avg_postprocessing_time = sum(x['time'] for x in performance_metrics['postprocessing_time']) / len(performance_metrics['postprocessing_time'])

        avg_inference_time = 0
        if performance_metrics['inference_time']:
            avg_inference_time = sum(x['time'] for x in performance_metrics['inference_time']) / len(performance_metrics['inference_time'])

        return jsonify({
            'total_detections': total_detections,
            'avg_confidence': avg_confidence,
            'detection_rate': detection_rate,
            'uptime': uptime_str,
            'fps': performance_metrics['fps'],
            'batch_processing_time': performance_metrics['batch_processing_time'][-200:],
            'batch_efficiency': performance_metrics['batch_efficiency'][-200:],
            'preprocessing_time': performance_metrics['preprocessing_time'][-200:],
            'postprocessing_time': performance_metrics['postprocessing_time'][-200:],
            'inference_time': performance_metrics['inference_time'][-200:],
            'memory_usage': performance_metrics['memory_usage'][-200:],
            'confidence': performance_metrics['confidence'][-200:],
            'detections': performance_metrics['detections'][-200:],
            'avg_batch_time': avg_batch_time,
            'avg_efficiency': avg_efficiency,
            'avg_preprocessing_time': avg_preprocessing_time,
            'avg_postprocessing_time': avg_postprocessing_time,
            'avg_inference_time': avg_inference_time
        })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting 4-camera grid viewer with BATCHED YOLOv11 detection...")
    print(f"Configuration from .env:")
    print(f"  NVR IP: {NVR_IP}")
    print(f"  YOLO Model: {YOLO_MODEL}")
    print(f"  Device: {YOLO_DEVICE.upper()}")
    print(f"  Batch Processing: ENABLED")
    print(f"  GPU Available: {USE_GPU}")
    print("Open browser: http://localhost:5001")
    print("="*60 + "\n")

    # Initialize performance metrics start time
    with metrics_lock:
        performance_metrics['start_time'] = time.time()

    # Start frame capture threads for all cameras
    for channel in cameras.keys():
        t = threading.Thread(target=frame_capture_thread, args=(channel,), daemon=True)
        t.start()
        print(f"[OK] Frame capture thread started for channel {channel}")

    # Start batch detection thread
    t_batch = threading.Thread(target=batch_detection_thread, daemon=True)
    t_batch.start()
    print("[OK] Batch detection thread started")

    print("\nAll systems ready!")
    print("Multi-user support enabled - Multiple users can connect simultaneously")
    print("Note: Each video stream runs in its own thread")

    # Run with increased thread support for multi-user
    # threaded=True allows multiple requests to be handled concurrently
    # processes=1 keeps it single-process (multiple threads, not processes)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True, processes=1)


