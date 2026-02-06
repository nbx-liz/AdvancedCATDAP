import subprocess
import time
import sys
import os
import platform

def launch():
    if platform.system() != "Windows":
        print("This script is designed for Windows.")
        return

    # Base command prefixes
    # We use 'start' to open new command prompt windows
    
    # 1. API
    print("Launching API...")
    subprocess.Popen('start cmd /k "set PYTHONPATH=. && uv run uvicorn advanced_catdap.service.api:app --reload --port 8000"', shell=True)
    
    # 2. Worker (Implicitly handled by JobManager)
    
    # 3. Frontend (Dash)
    print("Launching Frontend (Dash)...")
    # Give API a second to spin up
    time.sleep(2)
    # Set API_URL explicitly, though default is usually fine
    subprocess.Popen('start cmd /k "set PYTHONPATH=. && set API_URL=http://127.0.0.1:8000 && uv run python advanced_catdap/frontend/dash_app.py"', shell=True)

    print("\n--- Components Launched ---")
    print("1. API: http://localhost:8000")
    print("2. Frontend: http://localhost:8050")
    print("3. Job Worker: (Local Subprocesses)")

if __name__ == "__main__":
    launch()
