from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def current_git_commit(repo_root: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def build_run_record(
    *,
    event: str,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    split_id: str | None = None,
    command: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    data = config.get("data", {})
    model = config.get("model", {})
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "model": model.get("name"),
        "output": data.get("output"),
        "region": data.get("region"),
        "split_id": split_id,
        "config_fingerprint": config_fingerprint(config),
        "git_commit": git_commit,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "predictions_path": str(predictions_path) if predictions_path else None,
        "metrics": metrics or {},
        "command": command,
        "config": config,
    }


def append_run_record(
    log_path: str | Path,
    *,
    event: str,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    split_id: str | None = None,
    command: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    record = build_run_record(
        event=event,
        config=config,
        metrics=metrics,
        checkpoint_path=checkpoint_path,
        predictions_path=predictions_path,
        split_id=split_id,
        command=command,
        git_commit=git_commit,
    )
    output = Path(log_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def default_split_id(config: dict[str, Any]) -> str:
    data = config.get("data", {})
    seed = config.get("seed", "unknown")
    train_fraction = data.get("train_fraction", "unknown")
    test_patient_ids = data.get("test_patient_ids") or []
    if test_patient_ids:
        return f"explicit-test-{len(test_patient_ids)}-seed-{seed}"
    return f"random-train-{train_fraction}-seed-{seed}"
