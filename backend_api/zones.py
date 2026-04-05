import json
import time
from flask import Blueprint, request, jsonify
from .database import get_db

zones_bp = Blueprint('zones', __name__)


@zones_bp.route('/zones', methods=['GET'])
def list_zones():
    """List all zones/lines for all cameras."""
    conn = get_db()
    cursor = conn.execute("SELECT * FROM zones ORDER BY created_at DESC")
    zones = []
    for row in cursor.fetchall():
        zones.append({
            'id': row['id'],
            'camera': row['camera'],
            'type': row['type'],
            'label': row['label'],
            'coords': json.loads(row['coords']),
            'classes': json.loads(row['classes']) if row['classes'] else None,
            'color': row['color'],
            'enabled': bool(row['enabled']),
            'created_at': row['created_at'],
        })
    return jsonify({'zones': zones})


@zones_bp.route('/zones', methods=['POST'])
def create_zone():
    """Create a new zone."""
    data = request.get_json()
    zone_id = data.get('id', f"zone_{int(time.time() * 1000)}")

    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO zones (id, camera, type, label, coords, classes, color, enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            zone_id,
            data['camera'],
            data['type'],
            data.get('label', ''),
            json.dumps(data['coords']),
            json.dumps(data.get('classes')) if data.get('classes') else None,
            data.get('color', '#00ff88'),
            1 if data.get('enabled', True) else 0,
            time.time(),
        )
    )
    conn.commit()

    return jsonify({'id': zone_id, 'success': True})


@zones_bp.route('/zones/<zone_id>', methods=['PUT'])
def update_zone(zone_id):
    """Update an existing zone."""
    data = request.get_json()
    conn = get_db()

    updates = []
    params = []

    for field in ['camera', 'type', 'label', 'color', 'enabled']:
        if field in data:
            updates.append(f"{field} = ?")
            val = data[field]
            if field == 'enabled':
                val = 1 if val else 0
            params.append(val)

    if 'coords' in data:
        updates.append("coords = ?")
        params.append(json.dumps(data['coords']))

    if 'classes' in data:
        updates.append("classes = ?")
        params.append(json.dumps(data['classes']) if data['classes'] else None)

    if not updates:
        return jsonify({'error': 'No fields to update'}), 400

    params.append(zone_id)
    conn.execute(f"UPDATE zones SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()

    return jsonify({'success': True})


@zones_bp.route('/zones/<zone_id>', methods=['DELETE'])
def delete_zone(zone_id):
    """Delete a zone."""
    conn = get_db()
    conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
    conn.commit()
    return jsonify({'success': True})
