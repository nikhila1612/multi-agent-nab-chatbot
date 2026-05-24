"""
app.py — Flask Web Server for the NAB Multi-Agent Banking Chatbot
=================================================================
This file is the entry point for the backend server. It handles all incoming
HTTP requests from the frontend (browser) and passes them to the AI agent
system defined in agents.py.
 
How it works:
  1. A user types a message in the chat UI.
  2. The browser sends that message to this server via the /api/chat endpoint.
  3. This server forwards it to the OrchestratorAgent, which figures out which
     specialist AI agent (loans, payments, accounts, etc.) should answer.
  4. The answer is sent back to the browser as a JSON response.
"""

import os
import json
import uuid
import hashlib
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from agents import OrchestratorAgent

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Initialize orchestrator
orchestrator = OrchestratorAgent()

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Serve the chat interface.
    When a user navigates to the site (e.g. http://localhost:5000),
    they get the index.html file from the frontend folder.
    """
    return send_from_directory("../frontend", "index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint 
 
    Expects a JSON body like:
        {
            "session_id": "abc123",  
            "message":    "What home loans does NAB offer?"
        }
 
    Returns a JSON response like:
        {
            "session_id":  "abc123",
            "response":    "NAB offers several home loan options...",
            "agent":       "loans_agent",
            "status":      "answered",
            "metrics": {
                "response_time_ms": 1234,
                "tokens_in":  350,
                "tokens_out": 120,
                "total_tokens": 470,
                "cached": false
            }
        }
    """
    data = request.json
    # Use the session ID sent by the browser, or create a new one if it's the
    # user's first message. The session ID lets the AI remember recent context
    session_id = data.get("session_id") or str(uuid.uuid4())
    user_message = data.get("message", "").strip()
     
    # Don't process empty messages - return an error instead.
    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Time the entire processing pipeline so we can report wall-clock time
        # in case the agent doesn't report its own timing.
        t0 = time.time()
        result = orchestrator.process(session_id, user_message)
        wall_ms = int((time.time() - t0) * 1000)

        metrics = result.get("metrics", {})
        # Use agent-reported time if available, else wall clock
        response_time_ms = metrics.get("response_time_ms") or wall_ms
        
        # Send the agent's answer back to the browser.
        return jsonify({
            "session_id": session_id,
            "response": result["response"],
            "agent": result.get("agent"),
            "status": result.get("status", "answered"),
            "metrics": {
                "response_time_ms": response_time_ms,
                "tokens_in":  metrics.get("tokens_in", 0),
                "tokens_out": metrics.get("tokens_out", 0),
                "total_tokens": metrics.get("tokens_in", 0) + metrics.get("tokens_out", 0),
                "cached": metrics.get("cached", False),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/session/<session_id>", methods=["DELETE"])
def clear_session(session_id):
    """
    Delete a user's conversation history.
 
    Called when the user explicitly starts a fresh chat (e.g. clicks "New Conversation").
    Removes all stored message turns for this session so the AI starts with
    a blank slate.
    """
    orchestrator.clear_session(session_id)
    return jsonify({"ok": True})

@app.route("/api/session/<session_id>/delete", methods=["POST"])
def delete_session_on_close(session_id):
    """
    Clean up a session when the user closes the browser tab.
 
    Browsers can't send a DELETE request via navigator.sendBeacon (used for
    reliable tab-close callbacks), so this POST endpoint does the same thing.
    This keeps the sessions file tidy and avoids accumulating stale sessions.
    """
    orchestrator.clear_session(session_id)
    return jsonify({"ok": True})

@app.route("/api/health")
def health():
    """
    Health check endpoint.
 
    Useful for monitoring tools (or a quick sanity check in the browser) to
    verify that the server is running and all agents have been initialised.
 
    Returns something like:
        {"status": "ok", "agents": ["accounts_agent", "loans_agent", ...]}
    """
    return jsonify({"status": "ok", "agents": list(orchestrator.agents.keys())})

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
