import httpx
import time
import sys
import os

BASE_URL = "http://localhost:8000"

def check_health():
    print(f"Checking API at {BASE_URL}...")
    try:
        resp = httpx.get(f"{BASE_URL}/", timeout=2.0)
        resp.raise_for_status()
        print("API is UP.")
        return True
    except Exception as e:
        print(f"API is DOWN or Unreachable: {e}")
        return False

def run_diagnostic():
    if not check_health():
        print("CRITICAL: API is not running. Check the 'API' console window for errors.")
        return

    # Create dummy file
    dummy_csv = "diag_data.csv"
    with open(dummy_csv, "w") as f:
        f.write("A,B,target\n1,x,0\n2,y,1\n3,x,0\n4,y,1")

    try:
        # 1. Upload
        print("\nAttempting Upload...")
        with open(dummy_csv, "rb") as f:
            files = {"file": ("diag_data.csv", f, "text/csv")}
            resp = httpx.post(f"{BASE_URL}/datasets", files=files)
        
        if resp.status_code != 200:
            print(f"Upload FAILED: {resp.text}")
            return
        
        dataset_id = resp.json()["dataset_id"]
        print(f"Upload OK. ID: {dataset_id}")

        # 2. Preview
        print("Checking Preview...")
        resp = httpx.get(f"{BASE_URL}/datasets/{dataset_id}/preview")
        if resp.status_code != 200:
            print(f"Preview FAILED: {resp.text}")
            return
        print("Preview OK.")

        # 3. Submit Job
        print("Submitting Analysis Job...")
        params = {
            "target_col": "target",
            "max_bins": 5,
            "top_k": 10
        }
        resp = httpx.post(f"{BASE_URL}/jobs", params={"dataset_id": dataset_id}, json=params)
        
        if resp.status_code != 202:
            print(f"Job Submission FAILED: {resp.text}")
            return
            
        job_id = resp.json()["job_id"]
        print(f"Job Submitted. ID: {job_id}")

        # 4. Poll Status
        print("Polling Status (Max 10s)...")
        for i in range(10):
            resp = httpx.get(f"{BASE_URL}/jobs/{job_id}")
            status = resp.json().get("status")
            print(f"[{i+1}s] Status: {status}")
            
            if status == "SUCCESS":
                print("Job SUCCESS! Backend is working correctly.")
                return
            elif status == "FAILURE":
                print(f"Job FAILED: {resp.json().get('error')}")
                return
            
            time.sleep(1)
            
        print("TIMEOUT: Job stuck in Pending/Progress. Check if Worker is running and connected to Redis.")

    except Exception as e:
        print(f"Diagnostic Error: {e}")
    finally:
        if os.path.exists(dummy_csv):
            os.remove(dummy_csv)

if __name__ == "__main__":
    run_diagnostic()
