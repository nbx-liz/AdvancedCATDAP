from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from advanced_catdap.service.executor import LocalProcessExecutor


def test_executor_passes_env_to_subprocess(tmp_path):
    ex = LocalProcessExecutor()
    log_file = tmp_path / "logs" / "worker.log"
    with patch("advanced_catdap.service.executor.subprocess.Popen") as popen:
        ex.submit(["echo", "ok"], log_file, env={"CATDAP_X": "1"})
        assert popen.called
        kwargs = popen.call_args.kwargs
        assert "env" in kwargs
        assert kwargs["env"]["CATDAP_X"] == "1"


def test_executor_wraps_submit_failure(tmp_path):
    ex = LocalProcessExecutor()
    log_file = tmp_path / "logs" / "worker.log"
    with patch("advanced_catdap.service.executor.subprocess.Popen", side_effect=OSError("boom")):
        with pytest.raises(RuntimeError, match="Failed to submit local process"):
            ex.submit(["bad"], log_file)
