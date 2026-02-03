from advanced_catdap.service.job_manager import JobManager
import os
from pathlib import Path

def debug():
    manager = JobManager(data_dir="data")
    job_id = "f43616b0ed3bbc2a3a8b8a3ac2b551e9" # From user report
    
    print(f"Checking job {job_id}...")
    file_path = manager.jobs_dir / f"{job_id}.json"
    
    if file_path.exists():
        print(f"File exists. Size: {file_path.stat().st_size} bytes")
        print("Content:")
        try:
            with open(file_path, "r") as f:
                print(f.read())
        except Exception as e:
            print(f"Read error: {e}")
    else:
        print("File does not exist.")
        # Check for log
        log_path = manager.jobs_dir / f"{job_id}.log"
        if log_path.exists():
            print("Log file exists. Content:")
            with open(log_path, "r") as f:
                print(f.read())
        else:
            print("No log file either.")

    print("\nAttempting get_job_status...")
    try:
        status = manager.get_job_status(job_id)
        print("Status retrieved:", status)
    except Exception as e:
        print(f"get_job_status failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
