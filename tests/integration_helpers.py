import time
from pathlib import Path
from typing import Any, Dict, Set, Tuple


def wait_for_terminal_status(
    client_or_manager: Any,
    job_id: str,
    timeout_s: float = 30.0,
    interval_s: float = 0.2,
) -> Tuple[Dict[str, Any] | None, Set[str]]:
    """
    Poll job status until SUCCESS/FAILURE.

    Supports:
    - FastAPI TestClient (uses GET /jobs/{job_id})
    - JobManager-like objects (uses get_job_status(job_id))
    """
    deadline = time.time() + timeout_s
    seen: Set[str] = set()
    last: Dict[str, Any] | None = None

    while time.time() < deadline:
        if hasattr(client_or_manager, "get_job_status"):
            status = client_or_manager.get_job_status(job_id)
        else:
            resp = client_or_manager.get(f"/jobs/{job_id}")
            if resp.status_code != 200:
                raise AssertionError(f"Failed to get status: {resp.status_code} {resp.text}")
            status = resp.json()

        last = status
        st = status.get("status")
        if st:
            seen.add(st)
        if st in {"SUCCESS", "FAILURE"}:
            return status, seen
        time.sleep(interval_s)

    return last, seen


def assert_job_log_contains(job_id: str, expected_text: str, logs_dir: Path) -> str:
    log_file = logs_dir / f"{job_id}.log"
    if not log_file.exists():
        raise AssertionError(f"Log file not found: {log_file}")
    content = log_file.read_text(encoding="utf-8", errors="ignore")
    if expected_text not in content:
        raise AssertionError(f"Expected text not found in {log_file}: {expected_text}")
    return content

