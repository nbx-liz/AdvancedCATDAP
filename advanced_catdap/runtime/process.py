from __future__ import annotations

import socket
import time


def find_free_port() -> int:
    """Find an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Wait for local server readiness."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(("127.0.0.1", port))
                return True
        except ConnectionRefusedError:
            time.sleep(0.5)
    return False

