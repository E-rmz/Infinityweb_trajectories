"""Small helpers shared across pipeline modules."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    """Write `obj` to `path` atomically.

    Writes to a sibling tempfile then os.replace()'s into place, so a
    Ctrl+C mid-write can't leave a half-written JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
