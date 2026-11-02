"""Content-addressed cache helpers for released video preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = 2


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path | None, *, required: bool = False) -> dict[str, Any]:
    if path is None:
        if required:
            raise FileNotFoundError("A required resource path was not provided.")
        return {"path": None, "exists": False}
    path = Path(path).expanduser()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required resource file not found: {path}")
        return {"path": str(path.resolve()), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def directory_identity(
    root: str | Path | None,
    relative_files: tuple[str, ...] = (),
    *,
    required: bool = False,
) -> dict[str, Any]:
    if root is None:
        if required:
            raise FileNotFoundError("A required resource directory was not provided.")
        return {"path": None, "exists": False, "files": {}}
    root = Path(root).expanduser()
    if not root.is_dir():
        if required:
            raise FileNotFoundError(f"Required resource directory not found: {root}")
        return {"path": str(root.resolve()), "exists": False, "files": {}}
    files = {
        relative: file_identity(root / relative, required=False)
        for relative in relative_files
    }
    return {"path": str(root.resolve()), "exists": True, "files": files}


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def artifact_identities(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    identities = {}
    for item in paths:
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(f"Expected completed cache artifact is missing: {path}")
        identities[path.name] = file_identity(path, required=True)
    return identities


def validate_complete_manifest(
    manifest_path: str | Path,
    expected_fingerprint: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return False, "manifest is missing", None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"manifest is unreadable: {exc}", None
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False, "manifest schema version does not match", manifest
    if manifest.get("status") != "complete":
        return False, "manifest does not record a complete run", manifest
    if manifest.get("cache_fingerprint") != expected_fingerprint:
        return False, "input or preprocessing configuration changed", manifest
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False, "manifest has no completed artifact identities", manifest
    for identity in artifacts.values():
        path = identity.get("path") if isinstance(identity, dict) else None
        if not path or not Path(path).is_file():
            return False, f"cache artifact is missing: {path}", manifest
        current = file_identity(path, required=True)
        if current.get("size") != identity.get("size") or current.get("sha256") != identity.get(
            "sha256"
        ):
            return False, f"cache artifact content changed: {path}", manifest
    return True, "verified", manifest

