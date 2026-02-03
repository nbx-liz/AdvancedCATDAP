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
    
    # 3. Frontend
    print("Launching Frontend...")
    # Give API a second to spin up
    time.sleep(2)
    subprocess.Popen('start cmd /k "set PYTHONPATH=. && uv run streamlit run advanced_catdap/frontend/app.py"', shell=True)

    print("\n--- Components Launched ---")
    print("1. API: http://localhost:8000")
    print("2. Frontend: http://localhost:8501")
    print("3. Job Worker: (Local Subprocesses)")

if __name__ == "__main__":
    launch()
