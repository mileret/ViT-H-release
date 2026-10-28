"""Public G1 tracking API."""

from pathlib import Path
from typing import Any

TRACKER_PROTOCOL_VERSION = 4


def track_file(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .tracker import track_file as run

    return run(*args, **kwargs)


__all__ = ["TRACKER_PROTOCOL_VERSION", "track_file"]
