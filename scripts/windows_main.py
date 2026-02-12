"""Windows launcher wrapper for AdvancedCATDAP runtime entrypoint."""

from __future__ import annotations

import os
import sys

# Add project root to path for direct script execution.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from advanced_catdap.runtime.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
