#!/usr/bin/env python
"""
Convenience script to start both API and web servers
"""
import subprocess
import sys
import time
import signal
import os

def start_servers():
    """Start both API and web servers"""
    print("Starting N-gram Predictor Web Interface...")
    print("=" * 50)
    
    api_process = None
    web_process = None
    
    try:
        # Start API server
        print("Starting API server on http://localhost:8000...")
        api_process = subprocess.Popen([sys.executable, "api.py"])
        
        # Wait a moment for API to start
        time.sleep(2)
        
        # Start web server
        print("Starting web server on http://localhost:3000...")
        web_process = subprocess.Popen([sys.executable, "web_server.py"])
        
        print("\n" + "=" * 50)
        print("✅ Both servers are running!")
        print("🌐 Open your browser and go to: http://localhost:3000")
        print("📡 API documentation available at: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop both servers")
        print("=" * 50)
        
        # Wait for processes
        try:
            api_process.wait()
        except KeyboardInterrupt:
            pass
            
    except KeyboardInterrupt:
        print("\n\nShutting down servers...")
    except Exception as e:
        print(f"Error starting servers: {e}")
    finally:
        # Clean up processes
        if api_process:
            try:
                api_process.terminate()
                api_process.wait(timeout=5)
            except:
                api_process.kill()
        
        if web_process:
            try:
                web_process.terminate()
                web_process.wait(timeout=5)
            except:
                web_process.kill()
        
        print("Servers stopped.")

if __name__ == "__main__":
    start_servers() 