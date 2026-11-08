"""
Run the tracker's Re-ID (appearance) encoder with OpenVINO.

BoT-SORT's Re-ID compares how detected objects look from frame to frame. With
YOLO26 detectors, Ultralytics can't reuse the detector's own features (its head
is end-to-end), so it runs a separate classifier, yolo26n-cls.pt, on a crop of
every detected object, in PyTorch. This module runs that classifier with
OpenVINO instead, on a device chosen by inference_backend.py's benchmark.

The embedding is the same one Ultralytics computes: the classifier's backbone
output (the layer before its classification head), average-pooled, with the
same crop preprocessing.

It plugs in by replacing Ultralytics' ReID class (ultralytics.trackers.utils.reid
in current versions, ultralytics.trackers.bot_sort in older ones) with a factory
that returns this encoder for our model directory and defers to the original
class for anything else. A hook is used rather than pointing the tracker config
at the exported model directly, because Ultralytics would then run the encoder
on the detector's PyTorch device and could not target the benchmarked OpenVINO
device. If no hook point exists, install() returns False and tracking keeps
Ultralytics' default.
"""
from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REID_WEIGHTS = 'yolo26n-cls.pt'
REID_TRACKERS = {'botsort', 'tracktrack', 'deepocsort'}  # trackers that build Ultralytics' ReID encoder
CACHE_DIR = Path(__file__).resolve().parent / 'model_cache'

_shared_encoders: dict[tuple[str, str], 'OpenVINOReID'] = {}


class _GlobalPool(torch.nn.Module):
    """Average-pool a feature map to one vector per image, as Ultralytics' embed mode does."""

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)


class OpenVINOReID:
    """Drop-in replacement for ultralytics.trackers.bot_sort.ReID, backed by OpenVINO."""

    def __init__(self, model_dir: str | Path, ov_device: str):
        import openvino as ov
        from ultralytics.data.augment import classify_transforms

        model_dir = Path(model_dir)
        info = json.loads((model_dir / 'reid.json').read_text(encoding='utf-8'))
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(model_dir / info['xml']), ov_device,
                                           {'PERFORMANCE_HINT': 'LATENCY'})
        self.transforms = classify_transforms(info['imgsz'])
        self.device = ov_device

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray | None]:
        """Return one embedding per detection (None for an empty crop). dets rows are xywh + extras."""
        from ultralytics.utils.ops import xywh2xyxy
        from ultralytics.utils.plotting import save_one_box

        # Same crops as Ultralytics' ReID: BGR, padded slightly, then converted to RGB once.
        crops = [save_one_box(det, img, save=False, BGR=True) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        valid = [c.size > 0 for c in crops]
        if not any(valid):
            return [None] * len(crops)
        batch = torch.stack([self.transforms(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                             for c, keep in zip(crops, valid) if keep])
        rows = iter(self.compiled(batch.numpy())[0])
        return [np.asarray(next(rows), dtype=np.float32) if keep else None for keep in valid]


def export_encoder(weights: str = REID_WEIGHTS, force: bool = False) -> Path:
    """Export the classifier backbone + pooling to OpenVINO IR (dynamic batch). Returns the model directory.

    Re-exports when the source weights' contents change (their hash is stored in reid.json) or when forced.
    """
    import openvino as ov
    from ultralytics import YOLO

    from inference_backend import resolve_weights, weights_id

    stem = Path(weights).stem
    target = CACHE_DIR / f'{stem}_reid_openvino_model'
    source_id = weights_id(weights)
    if not force:
        try:
            if json.loads((target / 'reid.json').read_text(encoding='utf-8')).get('weights_id') == source_id:
                return target
        except (OSError, ValueError):
            pass

    print(f"[INFO] Exporting {weights} Re-ID encoder to OpenVINO (one-time)...")
    if target.exists():
        shutil.rmtree(target)
    cls_model = YOLO(str(resolve_weights(weights))).model.float().fuse().eval()  # fuse() matches Ultralytics' predict path
    layers = cls_model.model
    if type(layers[-1]).__name__ != 'Classify' or not all(layer.f == -1 for layer in layers):
        raise RuntimeError(f'{weights} is not a plain sequential classifier; cannot export a Re-ID encoder')
    imgsz = int(cls_model.args.get('imgsz', 224))
    encoder = torch.nn.Sequential(*layers[:-1], _GlobalPool()).eval()

    with torch.no_grad():
        ov_model = ov.convert_model(encoder, example_input=torch.zeros(1, 3, imgsz, imgsz),
                                    input=[ov.PartialShape([-1, 3, imgsz, imgsz])])
    target.mkdir(parents=True, exist_ok=True)
    xml = f'{stem}_reid.xml'
    ov.save_model(ov_model, target / xml, compress_to_fp16=True)
    info = {'xml': xml, 'imgsz': imgsz, 'source': weights, 'weights_id': source_id}
    (target / 'reid.json').write_text(json.dumps(info), encoding='utf-8')  # written last: marks a complete export
    return target


def _hook_module():
    """The module whose global `ReID` the trackers instantiate, or None for unknown Ultralytics versions."""
    import importlib

    for name in ('ultralytics.trackers.utils.reid', 'ultralytics.trackers.bot_sort'):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if callable(getattr(module, 'ReID', None)):
            return module
    return None


def _original_reid():
    module = _hook_module()
    current = getattr(module, 'ReID', None)
    return getattr(current, '_original', current)


def install(model_dir: str | Path, ov_device: str) -> bool:
    """Make the trackers use OpenVINOReID when their config names `model_dir`. Returns False if not possible."""
    module = _hook_module()
    if module is None:
        return False
    original = _original_reid()

    target = str(Path(model_dir).resolve())
    key = (target, ov_device)
    if key not in _shared_encoders:
        _shared_encoders[key] = OpenVINOReID(target, ov_device)  # compile now so failures surface at startup

    def reid_factory(model, *args, **kwargs):
        try:
            if str(Path(model).resolve()) == target:
                return _shared_encoders[key]  # stateless, so one instance serves every camera's tracker
        except (TypeError, OSError):
            pass
        return original(model, *args, **kwargs)

    reid_factory._original = original
    module.ReID = reid_factory
    return True


def write_tracker_config(base_yaml: str | Path, model_dir: str | Path) -> Path | None:
    """Copy the tracker config with `model:` pointing at the OpenVINO encoder.

    Returns None (keep base_yaml) unless the tracker uses Ultralytics' ReID encoder (BoT-SORT, TrackTrack,
    Deep OC-SORT) with Re-ID on and model: auto.
    """
    import yaml

    cfg = yaml.safe_load(Path(base_yaml).read_text(encoding='utf-8')) or {}
    if (cfg.get('tracker_type') not in REID_TRACKERS or not cfg.get('with_reid')
            or str(cfg.get('model', 'auto')) != 'auto'):
        return None
    cfg['model'] = str(Path(model_dir).resolve())
    out = CACHE_DIR / 'tracker_openvino_reid.yaml'
    header = (f"# Generated at startup from {Path(base_yaml).name} - edit that file instead.\n"
              f"# Same settings, with Re-ID running on OpenVINO (openvino_reid.py).\n")
    out.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return out


def benchmark(model_dir: str | Path, ov_devices: list[str], n_crops: int = 4) -> dict[str, float]:
    """Median ms to embed `n_crops` detections: Ultralytics' PyTorch ReID ('pytorch') vs OpenVINO per device."""
    from ultralytics.utils import LOGGER

    original = _original_reid()
    if original is None:
        raise RuntimeError('this Ultralytics version has no ReID class to compare against')
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    dets = np.array([[200 + 250 * i, 360, 120 + 20 * i, 220, i] for i in range(n_crops)], dtype=np.float32)

    level = LOGGER.level
    LOGGER.setLevel(40)
    try:
        candidates = {'pytorch': original(REID_WEIGHTS)}
    finally:
        LOGGER.setLevel(level)
    for d in ov_devices:
        try:
            candidates[f'intel:{d.lower()}'] = OpenVINOReID(model_dir, d)
        except Exception as e:  # noqa: BLE001 - a device that can't compile the encoder is skipped
            print(f"[INFO]   Re-ID on {d}: can't run ({type(e).__name__}), skipping")

    for enc in candidates.values():
        for _ in range(3):
            enc(img, dets)
    samples = {name: [] for name in candidates}
    for _ in range(3):  # interleaved, so throttling affects every candidate equally
        for name, enc in candidates.items():
            for _ in range(5):
                t = time.perf_counter()
                enc(img, dets)
                samples[name].append((time.perf_counter() - t) * 1000)
    return {name: round(statistics.median(s), 2) for name, s in samples.items()}
