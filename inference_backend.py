"""
Pick the fastest way to run YOLO on the PC the app is running on.

Order of preference when YOLO_DEVICE=auto (the default):
  1. An NVIDIA GPU (CUDA), when PyTorch can see one.
  2. Otherwise, if OpenVINO is installed, a short benchmark compares PyTorch on
     the CPU with OpenVINO on every Intel device it finds (integrated or Arc GPU,
     CPU, NPU). OpenVINO is used only when it is at least MIN_SPEEDUP faster.
  3. Otherwise PyTorch on the CPU.

The same benchmark times the tracker's Re-ID encoder (openvino_reid.py) and
moves it to OpenVINO only where that is faster too.

The benchmark runs once per machine. Its result is cached in
model_cache/backend_choice.json, keyed by hostname, hardware and library
versions, so the same checkout can move between PCs. Set YOLO_REBENCHMARK=1
to measure again.

YOLO_DEVICE can also force a backend:
  cpu                              PyTorch on the CPU
  cuda (or 0, cuda:0)              PyTorch on an NVIDIA GPU
  intel:gpu, intel:cpu, intel:npu  OpenVINO on that Intel device
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

CACHE_DIR = Path(__file__).resolve().parent / 'model_cache'
CHOICE_FILE = CACHE_DIR / 'backend_choice.json'

# OpenVINO must beat PyTorch-on-CPU by at least this factor to be chosen.
MIN_SPEEDUP = 1.15
# Fixed input size for exported models (YOLO's default).
EXPORT_IMGSZ = 640
# Frame size used for the benchmark: a common 16:9 camera sub-stream.
BENCH_FRAME_SHAPE = (720, 1280, 3)


@dataclass
class InferenceBackend:
    """How to load and run the detector on this machine."""

    kind: str            # 'pytorch' | 'openvino' | 'tensorrt'
    device: str          # passed to predict()/track(): 'cpu', 'cuda', 'intel:gpu', ...
    label: str           # human-readable, shown in the dashboard
    batch_weights: str   # weights for a batch of frames (batched mode)
    single_weights: str  # weights for one frame at a time (sequential mode, tracking)
    batch_size: int      # fixed batch size of batch_weights; 0 means any size works
    reason: str          # why this backend was chosen
    reid_device: str | None = None  # OpenVINO device for tracking Re-ID ('intel:gpu', ...); None = Ultralytics default
    reid_label: str = 'Ultralytics default (PyTorch)'

    @property
    def uses_cuda(self) -> bool:
        return self.device == 'cuda'

    def tracker_config(self, base_yaml: str) -> str:
        """Tracker config to use: base_yaml, or a generated copy that runs Re-ID on OpenVINO (see openvino_reid.py)."""
        if not self.reid_device:
            return base_yaml
        try:
            import openvino_reid
            model_dir = openvino_reid.export_encoder()
            generated = openvino_reid.write_tracker_config(base_yaml, model_dir)
            if generated is None:
                self.reid_label = 'Ultralytics default (tracker config does not use model: auto)'
            elif openvino_reid.install(model_dir, self.reid_device.split(':', 1)[1].upper()):
                self.reid_label = f"OpenVINO · {self.reid_device.split(':', 1)[1].upper().split('.')[0]}"
                return str(generated)
            else:
                print("[WARN] This Ultralytics version has no BoT-SORT ReID hook; Re-ID stays on PyTorch.")
        except Exception as e:  # noqa: BLE001 - tracking must work even if the OpenVINO encoder can't be set up
            print(f"[WARN] Could not set up OpenVINO Re-ID ({type(e).__name__}: {e}); Re-ID stays on PyTorch.")
        self.reid_device = None
        return base_yaml

    def load_models(self) -> tuple[YOLO, YOLO]:
        """Return (batch_model, single_model). They are the same object when the weights match."""
        batch_model = self._load(self.batch_weights)
        if self.single_weights == self.batch_weights:
            return batch_model, batch_model
        return batch_model, self._load(self.single_weights)

    def load_single_model(self) -> YOLO:
        """A fresh one-frame model, e.g. one per camera for tracking (each keeps its own tracker state)."""
        return self._load(self.single_weights)

    def predict_batch(self, model: YOLO, frames: list) -> list:
        """Run one forward pass over `frames`, padding to the model's fixed batch size if needed."""
        n = len(frames)
        if not self.batch_size or n == self.batch_size:
            return model(frames, device=self.device, verbose=False)
        results = []
        for start in range(0, n, self.batch_size):
            chunk = frames[start:start + self.batch_size]
            padded = chunk + [chunk[-1]] * (self.batch_size - len(chunk))
            results.extend(model(padded, device=self.device, verbose=False)[:len(chunk)])
        return results

    def to_dict(self) -> dict:
        return {'kind': self.kind, 'device': self.device, 'label': self.label, 'reason': self.reason,
                'reid': self.reid_label}

    def _load(self, weights: str) -> YOLO:
        model = YOLO(weights, task='detect')
        if self.kind == 'pytorch':
            model.to(self.device)
        return model


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #

def select_backend(model_name: str, batch_size: int = 4) -> InferenceBackend:
    """Choose a backend from YOLO_DEVICE (default 'auto'). Never raises: falls back to PyTorch on the CPU."""
    requested = (os.getenv('YOLO_DEVICE') or 'auto').strip().lower()
    try:
        if requested == 'cpu':
            if _import_openvino() is not None:
                print("[INFO] YOLO_DEVICE=cpu forces PyTorch on the CPU. "
                      "Set YOLO_DEVICE=auto to let the app pick a faster backend if there is one.")
            return cpu_backend(model_name, 'YOLO_DEVICE=cpu in .env')

        if requested in ('cuda', 'gpu') or requested.startswith('cuda:') or requested.isdigit():
            if torch.cuda.is_available():
                return _cuda_backend(model_name, batch_size, f'YOLO_DEVICE={requested} in .env')
            print(f"[WARN] YOLO_DEVICE={requested}, but PyTorch can't see an NVIDIA GPU. Choosing automatically.")
            return _auto(model_name, batch_size)

        if requested.startswith('intel:'):
            backend = _forced_openvino(model_name, batch_size, requested)
            return backend if backend is not None else _auto(model_name, batch_size)

        if requested != 'auto':
            print(f"[WARN] Unknown YOLO_DEVICE={requested!r}. Choosing automatically.")
        return _auto(model_name, batch_size)
    except Exception as e:  # noqa: BLE001 - never block startup on backend selection
        print(f"[WARN] Backend selection failed ({type(e).__name__}: {e}). Using PyTorch on the CPU.")
        return cpu_backend(model_name, 'fallback after backend selection failed')


def cpu_backend(model_name: str, reason: str) -> InferenceBackend:
    return InferenceBackend('pytorch', 'cpu', 'PyTorch · CPU', model_name, model_name, 0, reason)


# --------------------------------------------------------------------------- #
# Automatic selection
# --------------------------------------------------------------------------- #

def _auto(model_name: str, batch_size: int) -> InferenceBackend:
    if torch.cuda.is_available():
        return _cuda_backend(model_name, batch_size, f'NVIDIA GPU found ({torch.cuda.get_device_name(0)})')

    ov = _import_openvino()
    if ov is None:
        print("[INFO] OpenVINO is not installed. On Intel PCs, `pip install openvino` "
              "can make detection several times faster.")
        return cpu_backend(model_name, 'no NVIDIA GPU, and OpenVINO is not installed')

    core = ov.Core()
    ov_devices = {d: _clean_name(core.get_property(d, 'FULL_DEVICE_NAME'))
                  for d in core.available_devices if d.split('.')[0] in ('CPU', 'GPU', 'NPU')}
    if not ov_devices:
        return cpu_backend(model_name, 'OpenVINO found no usable devices')

    fingerprint = {
        'host': platform.node(),
        'processor': platform.processor(),
        'openvino_devices': ov_devices,
        'versions': {'openvino': ov.__version__.split('-')[0], 'ultralytics': ultralytics.__version__,
                     'torch': torch.__version__},
        'model': Path(model_name).name,
        'batch_size': batch_size,
    }
    key = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]

    cache = _read_cache()
    entry = cache.get(key)
    rebenchmark = os.getenv('YOLO_REBENCHMARK', '').strip().lower() in ('1', 'true', 'yes')
    changed = False
    if entry and not rebenchmark:
        print(f"[INFO] Using cached backend benchmark from {entry['measured_at']} "
              f"(set YOLO_REBENCHMARK=1 to measure again)")
    else:
        entry = _benchmark(model_name, batch_size, ov_devices)
        entry['fingerprint'] = fingerprint
        changed = True
    if 'reid' not in entry:
        reid = _benchmark_reid(ov_devices)
        if reid is not None:  # not cached on failure, so it is retried next start
            entry['reid'] = reid
            changed = True
    if changed:
        cache[key] = entry
        _write_cache(cache)

    timings = entry['timings_ms']
    summary = ', '.join(f"{_device_title(d, ov_devices)} {ms:.0f} ms" for d, ms in sorted(timings.items(), key=lambda kv: kv[1]))
    choice = entry['choice']
    if choice == 'cpu':
        backend = cpu_backend(model_name, f'fastest on this PC: {summary} per {batch_size} frames')
    else:
        backend = _openvino_backend(model_name, batch_size, choice, ov_devices,
                                    f'fastest on this PC: {summary} per {batch_size} frames')
    reid_choice = (entry.get('reid') or {}).get('choice', 'pytorch')
    if reid_choice != 'pytorch':
        backend.reid_device = reid_choice
    return backend


def _benchmark_reid(ov_devices: dict) -> dict | None:
    """Time the tracker's Re-ID encoder (1 and 4 objects): Ultralytics' PyTorch path vs OpenVINO per device."""
    try:
        import openvino_reid
        model_dir = openvino_reid.export_encoder()
        one = openvino_reid.benchmark(model_dir, list(ov_devices), n_crops=1)
        four = openvino_reid.benchmark(model_dir, list(ov_devices), n_crops=4)
    except Exception as e:  # noqa: BLE001 - Re-ID acceleration is optional
        print(f"[INFO] Re-ID benchmark skipped ({type(e).__name__}: {e}); tracking Re-ID stays on PyTorch.")
        return None
    timings = {name: round(one[name] + four[name], 1) for name in one if name in four}
    choice = 'pytorch'
    ov_timings = {k: v for k, v in timings.items() if k != 'pytorch'}
    if ov_timings and 'pytorch' in timings:
        best = min(ov_timings, key=ov_timings.get)
        if timings['pytorch'] / ov_timings[best] >= MIN_SPEEDUP:
            choice = best
    print("[INFO] Tracking Re-ID encoder (ms for 1 + 4 objects):")
    for name, ms in sorted(timings.items(), key=lambda kv: kv[1]):
        title = 'PyTorch (Ultralytics default)' if name == 'pytorch' else _device_title(name, ov_devices)
        print(f"[INFO]   {title:45s} {ms:7.1f} ms{'  <- chosen' if name == choice else ''}")
    return {'choice': choice, 'timings_ms': timings, 'crops': [1, 4]}


def _benchmark(model_name: str, batch_size: int, ov_devices: dict) -> dict:
    """Time PyTorch-CPU against OpenVINO on each Intel device; return the cache entry."""
    print(f"[INFO] First run on this PC with {Path(model_name).name}: benchmarking backends "
          f"(about a minute, cached afterwards)...")
    _export_openvino(model_name, 1)  # used later for sequential mode and tracking
    bn = _export_openvino(model_name, batch_size)

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, BENCH_FRAME_SHAPE, dtype=np.uint8) for _ in range(batch_size)]

    candidates = {'cpu': (YOLO(model_name, task='detect'), 'cpu')}
    for d in ov_devices:
        candidates[f'intel:{d.lower()}'] = (YOLO(bn, task='detect'), f'intel:{d.lower()}')

    # Warm up (the first OpenVINO call also compiles the model for its device).
    usable = {}
    for name, (model, device) in candidates.items():
        try:
            for _ in range(2):
                model(frames, device=device, verbose=False)
            usable[name] = (model, device)
        except Exception as e:  # noqa: BLE001 - a device that can't run the model is just skipped
            print(f"[INFO]   {name}: can't run this model ({type(e).__name__}), skipping")

    # Interleave rounds so thermal throttling affects every candidate equally.
    samples = {name: [] for name in usable}
    for _ in range(3):
        for name, (model, device) in usable.items():
            for _ in range(3):
                t = time.perf_counter()
                model(frames, device=device, verbose=False)
                samples[name].append((time.perf_counter() - t) * 1000)
    timings = {name: round(statistics.median(s), 1) for name, s in samples.items()}

    choice = 'cpu'
    ov_timings = {k: v for k, v in timings.items() if k != 'cpu'}
    if ov_timings and 'cpu' in timings:
        best = min(ov_timings, key=ov_timings.get)
        if timings['cpu'] / ov_timings[best] >= MIN_SPEEDUP:
            choice = best
    elif ov_timings:  # PyTorch itself failed; take the best OpenVINO device
        choice = min(ov_timings, key=ov_timings.get)

    for name, ms in sorted(timings.items(), key=lambda kv: kv[1]):
        marker = '  <- chosen' if name == choice else ''
        print(f"[INFO]   {_device_title(name, ov_devices):45s} {ms:7.1f} ms per {batch_size} frames{marker}")
    return {'choice': choice, 'timings_ms': timings, 'measured_at': datetime.now().isoformat(timespec='seconds')}


# --------------------------------------------------------------------------- #
# Backend builders
# --------------------------------------------------------------------------- #

def _cuda_backend(model_name: str, batch_size: int, reason: str) -> InferenceBackend:
    label = f'PyTorch · {torch.cuda.get_device_name(0)}'
    engine = f"{Path(model_name).stem}_batch{batch_size}.engine"
    if os.path.exists(engine):
        # A TensorRT engine exported with dynamic=True accepts any batch up to batch_size.
        return InferenceBackend('tensorrt', 'cuda', f'TensorRT · {torch.cuda.get_device_name(0)}',
                                engine, model_name, 0, f'{reason}; using TensorRT engine {engine}')
    return InferenceBackend('pytorch', 'cuda', label, model_name, model_name, 0, reason)


def _openvino_backend(model_name: str, batch_size: int, device: str, ov_devices: dict, reason: str) -> InferenceBackend:
    return InferenceBackend(
        kind='openvino',
        device=device,
        label=f"OpenVINO · {_device_title(device, ov_devices, short=True)}",
        batch_weights=_export_openvino(model_name, batch_size),
        single_weights=_export_openvino(model_name, 1),
        batch_size=batch_size,
        reason=reason,
    )


def _forced_openvino(model_name: str, batch_size: int, requested: str) -> InferenceBackend | None:
    ov = _import_openvino()
    if ov is None:
        print(f"[WARN] YOLO_DEVICE={requested} needs OpenVINO (pip install openvino). Choosing automatically.")
        return None
    core = ov.Core()
    wanted = requested.split(':', 1)[1].upper()
    if wanted not in core.available_devices:
        print(f"[WARN] OpenVINO has no {wanted} device here (found {', '.join(core.available_devices)}). "
              f"Choosing automatically.")
        return None
    ov_devices = {wanted: _clean_name(core.get_property(wanted, 'FULL_DEVICE_NAME'))}
    backend = _openvino_backend(model_name, batch_size, requested, ov_devices, f'YOLO_DEVICE={requested} in .env')
    backend.reid_device = requested  # forced: Re-ID follows the detector's device
    return backend


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _export_openvino(model_name: str, batch: int) -> str:
    """Export model_name to an OpenVINO IR with a fixed batch size, once. Returns the model directory."""
    # Ultralytics recognises OpenVINO models by the '_openvino_model' folder suffix.
    target = CACHE_DIR / f"{Path(model_name).stem}_b{batch}_openvino_model"
    if _exported_with_current_ultralytics(target):
        return str(target)
    print(f"[INFO] Exporting {Path(model_name).name} to OpenVINO (batch {batch}, one-time)...")
    CACHE_DIR.mkdir(exist_ok=True)
    from ultralytics.cfg import DEFAULT_CFG_DICT
    fp16 = {'quantize': 16} if 'quantize' in DEFAULT_CFG_DICT else {'half': True}  # 'half' is deprecated in newer versions
    exported = YOLO(model_name).export(format='openvino', batch=batch, imgsz=EXPORT_IMGSZ,
                                       dynamic=False, verbose=False, **fp16)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(exported), str(target))
    return str(target)


def _exported_with_current_ultralytics(model_dir: Path) -> bool:
    """True if model_dir holds an export made by the installed Ultralytics version (re-export after upgrades)."""
    try:
        import yaml
        meta = yaml.safe_load((model_dir / 'metadata.yaml').read_text(encoding='utf-8')) or {}
    except (OSError, ValueError):
        return False
    return str(meta.get('version')) == ultralytics.__version__


def _import_openvino():
    try:
        import openvino as ov
        return ov
    except ImportError:
        return None


def _clean_name(name: str) -> str:
    return ' '.join(name.replace('(R)', '').replace('(TM)', '').split())


def _device_title(device: str, ov_devices: dict, short: bool = False) -> str:
    if device == 'cpu':
        return 'PyTorch CPU'
    ov_name = device.split(':', 1)[1].upper()
    hw = ov_devices.get(ov_name, ov_name)
    return hw if short else f'OpenVINO {ov_name.split(".")[0]} ({hw})'


def _read_cache() -> dict:
    try:
        return json.loads(CHOICE_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _write_cache(cache: dict) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        CHOICE_FILE.write_text(json.dumps(cache, indent=2), encoding='utf-8')
    except OSError as e:
        print(f"[WARN] Could not save backend benchmark to {CHOICE_FILE}: {e}")
