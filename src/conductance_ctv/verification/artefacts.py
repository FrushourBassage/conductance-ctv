"""The three release artefacts and the integrity manifest.

The order matters and is enforced by the driver: the claim map, the verification
report and the plain-text summary are all written before the manifest, so that
the manifest digests a tree that already contains them. The manifest excludes
itself and nothing else, and its own digests are recomputed from the live tree
before a run is declared finished.

Ref: the verification pass described in the project context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conductance_ctv.utils.config import repo_root
from conductance_ctv.utils.io import sha256_file, write_json, write_text

MANIFEST_NAME = "integrity_manifest.json"
CLAIM_MAP_NAME = "claim_to_code.json"
REPORT_NAME = "verification_report.json"
SUMMARY_NAME = "verification_summary.txt"
DATASET_URLS_NAME = "dataset_urls.txt"

SKIPPED_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}
SKIPPED_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".tmp"}
SKIPPED_NAMES = {".DS_Store"}


@dataclass
class ManifestEntry:
    path: str
    bytes: int
    sha256: str

    def record(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass
class IntegrityManifest:
    root: str
    files: list[ManifestEntry] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)

    @property
    def digest(self) -> str:
        import hashlib

        accumulator = hashlib.sha256()
        for entry in sorted(self.files, key=lambda item: item.path):
            accumulator.update(entry.path.encode("utf-8"))
            accumulator.update(entry.sha256.encode("utf-8"))
        return accumulator.hexdigest()

    def record(self) -> dict[str, Any]:
        return {
            "root": ".",
            "file_count": len(self.files),
            "total_bytes": sum(entry.bytes for entry in self.files),
            "excluded": self.excluded,
            "manifest_digest": self.digest,
            "files": [entry.record() for entry in self.files],
        }


def iter_tracked_files(root: Path, skip: set[str]) -> list[Path]:
    """Every file of the tree except the skipped directories, suffixes and names."""
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIPPED_DIRECTORIES for part in path.parts):
            continue
        if path.suffix in SKIPPED_SUFFIXES or path.name in SKIPPED_NAMES:
            continue
        if path.name in skip:
            continue
        out.append(path)
    return out


def build_manifest(
    root: str | Path | None = None,
    exclude: set[str] | None = None,
    extra_entries: list[ManifestEntry] | None = None,
) -> IntegrityManifest:
    """Hash the tree, excluding the manifest itself and any named file."""
    base = repo_root() if root is None else Path(root)
    skip = {MANIFEST_NAME} if exclude is None else set(exclude) | {MANIFEST_NAME}
    manifest = IntegrityManifest(root=str(base), excluded=sorted(skip))
    for path in iter_tracked_files(base, skip):
        manifest.files.append(
            ManifestEntry(
                path=str(path.relative_to(base)),
                bytes=int(path.stat().st_size),
                sha256=sha256_file(path),
            )
        )
    if extra_entries:
        manifest.files.extend(extra_entries)
    return manifest


def write_claim_map(payload: dict[str, Any], root: str | Path | None = None) -> Path:
    base = repo_root() if root is None else Path(root)
    return write_json(base / CLAIM_MAP_NAME, payload)


def write_verification_report(payload: dict[str, Any], root: str | Path | None = None) -> Path:
    base = repo_root() if root is None else Path(root)
    return write_json(base / REPORT_NAME, payload)


def write_summary(lines: list[str], root: str | Path | None = None) -> Path:
    base = repo_root() if root is None else Path(root)
    return write_text(base / SUMMARY_NAME, "\n".join(lines).rstrip("\n") + "\n")


def write_dataset_urls(entries: list[dict[str, str]], root: str | Path | None = None) -> Path:
    """Write the link list, keeping only entries whose content was checked."""
    base = repo_root() if root is None else Path(root)
    lines = [
        "# Dataset and resource links, with the version, licence and what was checked.",
        "# Rewritten by `python -m conductance_ctv.cli.verify`; only entries whose live",
        "# content matched the expected resource are listed.",
        "",
    ]
    for entry in entries:
        lines.append(f"name: {entry['name']}")
        lines.append(f"version: {entry['version']}")
        lines.append(f"licence: {entry['licence']}")
        lines.append(f"role: {entry['role']}")
        lines.append(f"url: {entry['url']}")
        if entry.get("recorded_doi") and entry["recorded_doi"] != entry["url"]:
            lines.append(f"recorded_doi: {entry['recorded_doi']}")
        lines.append(f"checked: {entry['checked']}")
        lines.append("")
    return write_text(base / DATASET_URLS_NAME, "\n".join(lines).rstrip("\n") + "\n")


def write_integrity_manifest(root: str | Path | None = None) -> Path:
    """Write the manifest last, so it digests a tree that already holds every artefact."""
    base = repo_root() if root is None else Path(root)
    manifest = build_manifest(base)
    return write_json(base / MANIFEST_NAME, manifest.record())


def verify_manifest(root: str | Path | None = None) -> dict[str, Any]:
    """Re-hash the live tree and diff it against the shipped manifest."""
    base = repo_root() if root is None else Path(root)
    shipped = __import__("json").loads((base / MANIFEST_NAME).read_text(encoding="utf-8"))
    live = build_manifest(base)
    shipped_map = {entry["path"]: entry["sha256"] for entry in shipped.get("files", [])}
    live_map = {entry.path: entry.sha256 for entry in live.files}
    missing = sorted(set(shipped_map) - set(live_map))
    added = sorted(set(live_map) - set(shipped_map))
    changed = sorted(path for path in set(shipped_map) & set(live_map) if shipped_map[path] != live_map[path])
    return {
        "manifest_digest": shipped.get("manifest_digest"),
        "live_digest": live.digest,
        "matches": not missing and not added and not changed,
        "missing": missing[:20],
        "added": added[:20],
        "changed": changed[:20],
        "files": len(live_map),
    }


def artefact_paths(root: str | Path | None = None) -> dict[str, Path]:
    base = repo_root() if root is None else Path(root)
    return {
        "claim_map": base / CLAIM_MAP_NAME,
        "verification_report": base / REPORT_NAME,
        "summary": base / SUMMARY_NAME,
        "integrity_manifest": base / MANIFEST_NAME,
        "dataset_urls": base / DATASET_URLS_NAME,
    }
