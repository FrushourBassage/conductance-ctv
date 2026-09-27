"""Atomic writers for the textual and binary artefacts this project emits.

Every report the project produces is written through here so that a partially
written file is never observable, and so that artefact permissions stay readable
by the whole group rather than inheriting the private mode of a temp file.

Ref: Sec. 6.11 (reporting and reproducibility).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_FILE_MODE = 0o644


def write_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomic binary write with world-readable permissions."""
    return _atomic_bytes(Path(path), payload)


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, _FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_text(path: str | Path, text: str) -> Path:
    return _atomic_bytes(Path(path), text.encode("utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any], indent: int = 2) -> Path:
    body = json.dumps(payload, indent=indent, sort_keys=False, ensure_ascii=False, default=_json_default)
    return _atomic_bytes(Path(path), (body + "\n").encode("utf-8"))


def write_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(target.parent), suffix=".npz", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        # numpy's stubs mistype the keyword form of savez_compressed.
        savez: Any = np.savez_compressed
        savez(tmp, **arrays)
        os.chmod(tmp, _FILE_MODE)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")
