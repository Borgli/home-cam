import json
from flask import Blueprint, request, jsonify

classes_bp = Blueprint('classes', __name__)

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Default: all classes enabled
enabled_classes = set(range(len(COCO_CLASSES)))


def get_enabled_classes():
    """Get currently enabled class IDs."""
    return enabled_classes


@classes_bp.route('/classes', methods=['GET'])
def list_classes():
    """List all COCO classes with enabled status."""
    classes = []
    for i, name in enumerate(COCO_CLASSES):
        classes.append({
            'id': i,
            'name': name,
            'enabled': i in enabled_classes,
        })
    return jsonify({'classes': classes})


@classes_bp.route('/classes/filter', methods=['POST'])
def set_class_filter():
    """Set which classes to detect."""
    global enabled_classes
    data = request.get_json()
    class_ids = data.get('classes', [])

    enabled_classes = set(int(c) for c in class_ids if 0 <= int(c) < len(COCO_CLASSES))

    return jsonify({
        'enabled_count': len(enabled_classes),
        'total': len(COCO_CLASSES),
        'success': True,
    })
