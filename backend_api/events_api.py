import time
from flask import Blueprint, request, jsonify
from .database import get_db, query_db

events_bp = Blueprint('events', __name__)


@events_bp.route('/events', methods=['GET'])
def list_events():
    """Get paginated event log."""
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 100))
    camera = request.args.get('camera')
    class_name = request.args.get('class')
    offset = (page - 1) * per_page

    conditions = []
    params = []

    if camera is not None:
        conditions.append("channel = ?")
        params.append(int(camera))

    if class_name:
        conditions.append("class_name = ?")
        params.append(class_name)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    conn = get_db()

    # Total count
    count = conn.execute(f"SELECT COUNT(*) FROM detections {where}", params).fetchone()[0]

    # Paginated results
    rows = conn.execute(
        f"SELECT * FROM detections {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        params + [per_page, offset]
    ).fetchall()

    events = [dict(row) for row in rows]

    return jsonify({
        'events': events,
        'total': count,
        'page': page,
        'per_page': per_page,
        'pages': (count + per_page - 1) // per_page,
    })


@events_bp.route('/events/stats', methods=['GET'])
def event_stats():
    """Get summary statistics."""
    conn = get_db()

    total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    unique_classes = conn.execute("SELECT COUNT(DISTINCT class_name) FROM detections").fetchone()[0]

    # Per-class counts
    class_counts = conn.execute(
        "SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name ORDER BY count DESC LIMIT 20"
    ).fetchall()

    # Per-camera counts
    camera_counts = conn.execute(
        "SELECT channel, COUNT(*) as count FROM detections GROUP BY channel ORDER BY channel"
    ).fetchall()

    # Hourly distribution (last 24h)
    day_ago = time.time() - 86400
    hourly = conn.execute(
        "SELECT CAST((timestamp - ?) / 3600 AS INTEGER) as hour, COUNT(*) as count FROM detections WHERE timestamp > ? GROUP BY hour ORDER BY hour",
        (day_ago, day_ago)
    ).fetchall()

    return jsonify({
        'total_detections': total,
        'unique_classes': unique_classes,
        'class_counts': [{'class': r['class_name'], 'count': r['count']} for r in class_counts],
        'camera_counts': [{'camera': r['channel'], 'count': r['count']} for r in camera_counts],
        'hourly': [{'hour': r['hour'], 'count': r['count']} for r in hourly],
    })


@events_bp.route('/db/query', methods=['POST'])
def db_query():
    """Execute a read-only SQL query."""
    data = request.get_json()
    sql = data.get('sql', '')

    try:
        rows = query_db(sql)
        return jsonify({'rows': rows, 'count': len(rows)})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f"SQL Error: {str(e)}"}), 400


@events_bp.route('/db/schema', methods=['GET'])
def db_schema():
    """Get database schema for LLM context."""
    from .database import get_schema_info
    return jsonify({'schema': get_schema_info()})
