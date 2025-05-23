#!/usr/bin/env python
"""
Simple web server for serving the frontend interface
"""
import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 3000
WEB_DIR = "web"

class WebHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/':
            self.path = '/templates/index.html'
        super().do_GET()

def start_web_server():
    """Start the web server"""
    # Check if web directory exists
    if not os.path.exists(WEB_DIR):
        print(f"Error: Web directory '{WEB_DIR}' not found!")
        sys.exit(1)
    
    # Check if required files exist
    index_file = os.path.join(WEB_DIR, "templates", "index.html")
    if not os.path.exists(index_file):
        print(f"Error: {index_file} not found!")
        sys.exit(1)
    
    try:
        with socketserver.TCPServer(("", PORT), WebHandler) as httpd:
            print(f"Web server starting on http://localhost:{PORT}")
            print("Serving frontend from web/ directory")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Error: Port {PORT} is already in use!")
            print("Please stop any running server on this port or change the PORT variable")
        else:
            print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_web_server() 