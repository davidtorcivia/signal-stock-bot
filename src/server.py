"""
Flask webhook server for receiving Signal messages.
"""

import asyncio
import logging
from flask import Flask, request, jsonify

from .signal.handler import SignalHandler

logger = logging.getLogger(__name__)


def create_app(signal_handler: SignalHandler) -> Flask:
    """
    Create Flask application with webhook endpoint.
    
    Args:
        signal_handler: Handler for processing Signal messages
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    
    # Store handler reference
    app.signal_handler = signal_handler
    
    @app.route("/webhook", methods=["POST"])
    def webhook():
        """
        Webhook endpoint for signal-cli-rest-api.
        
        Receives JSON payload with message data and dispatches
        to the signal handler for processing.
        """
        data = request.json
        
        if not data:
            logger.warning("Received empty webhook payload")
            return jsonify({"status": "error", "message": "Empty payload"}), 400
        
        logger.debug(f"Received webhook: {data}")
        
        # Run async handler in event loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(signal_handler.handle_webhook(data))
        except Exception as e:
            logger.exception(f"Error handling webhook: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
        finally:
            loop.close()
        
        return jsonify({"status": "ok"})
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        return jsonify({"status": "healthy"})
    
    @app.route("/", methods=["GET"])
    def index():
        """Root endpoint with basic info"""
        return jsonify({
            "name": "Signal Stock Bot",
            "status": "running",
            "endpoints": {
                "/webhook": "POST - Signal message webhook",
                "/health": "GET - Health check",
            }
        })
    
    return app
