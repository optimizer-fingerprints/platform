from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_FINGERPRINT_DIR = Path("fingerprints")
DEFAULT_INDEX_PATH = Path("web/public/fingerprints.json")
DEFAULT_TRACE_DIR = Path("logs/traces")


def stable_json_hash(payload: dict[str, Any], length: int = 10) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def fingerprint_id(*, world_id: str, optimizer_name: str, seed: int, probe: dict[str, Any], optimizer: dict[str, Any]) -> str:
    hash_payload = {
        "world_id": world_id,
        "optimizer": optimizer,
        "seed": seed,
        "probe": probe,
    }
    digest = stable_json_hash(hash_payload)
    return f"{world_id}__{optimizer_name}__seed{seed}__{digest}"


def fingerprint_path(root: Path, fingerprint: dict[str, Any]) -> Path:
    world_id = fingerprint["world"]["world_id"]
    optimizer_name = fingerprint["optimizer"]["name"]
    return root / world_id / optimizer_name / f"{fingerprint['fingerprint_id']}.json"


def write_fingerprint(root: Path, fingerprint: dict[str, Any]) -> Path:
    path = fingerprint_path(root, fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n")
    return path


def rebuild_index(fingerprint_root: Path, index_path: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(fingerprint_root.glob("*/*/*.json")):
        fingerprint = json.loads(path.read_text())
        entries.append(
            {
                "fingerprint_id": fingerprint["fingerprint_id"],
                "schema_version": fingerprint["schema_version"],
                "world_id": fingerprint["world"]["world_id"],
                "optimizer": fingerprint["optimizer"]["name"],
                "optimizer_family": fingerprint["optimizer"]["family"],
                "seed": fingerprint["world"]["seed"],
                "feature_count": len(fingerprint["feature_names"]),
                "path": path.as_posix(),
            }
        )

    index = {
        "schema_version": "fingerprint_index_v1",
        "fingerprint_root": fingerprint_root.as_posix(),
        "fingerprints": entries,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index
