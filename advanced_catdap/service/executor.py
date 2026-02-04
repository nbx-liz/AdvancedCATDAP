
import subprocess
import os
import sys
from typing import List, Dict, Protocol, Optional
from pathlib import Path

class JobExecutor(Protocol):
    def submit(self, cmd: List[str], log_file: Path, env: Optional[Dict[str, str]] = None) -> None:
        """
        Submits a job for execution.
        Raises exception if submission fails.
        """
        ...

class LocalProcessExecutor:
    """
    Executes jobs as local subprocesses.
    """
    def submit(self, cmd: List[str], log_file: Path, env: Optional[Dict[str, str]] = None) -> None:
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
            
        try:
            with open(log_file, "w") as f:
                # Popen is non-blocking
                subprocess.Popen(cmd, stdout=f, stderr=f, env=run_env)
        except Exception as e:
            raise RuntimeError(f"Failed to submit local process: {e}") from e
