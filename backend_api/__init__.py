from flask import Blueprint

from .database import init_db, get_db
from .zones import zones_bp
from .events_api import events_bp
from .classes_api import classes_bp
from .llm_service import llm_bp


def register_blueprints(app):
    """Register all API blueprints with the Flask app."""
    init_db()
    app.register_blueprint(zones_bp, url_prefix='/api')
    app.register_blueprint(events_bp, url_prefix='/api')
    app.register_blueprint(classes_bp, url_prefix='/api')
    app.register_blueprint(llm_bp, url_prefix='/api')

    # Debug: print registered routes
    for rule in app.url_map.iter_rules():
        if '/api/' in str(rule) and 'static' not in str(rule):
            print(f"  [ROUTE] {rule.methods} {rule.rule}")
