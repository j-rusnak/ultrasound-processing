import json
from pathlib import Path

from ultrasound_processing.tracking import append_run_record, config_fingerprint


def test_append_run_record_writes_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "runs.jsonl"
    config = {"model": {"name": "unet"}, "data": {"output": "FM", "region": "BAQ"}}

    record = append_run_record(
        log_path,
        event="evaluate",
        config=config,
        metrics={"mae": 0.1},
        checkpoint_path="artifacts/checkpoints/model.pt",
        predictions_path="artifacts/results/predictions.csv",
        split_id="seed-42",
        command="python scripts/evaluate.py",
    )

    line = log_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["event"] == "evaluate"
    assert parsed["model"] == "unet"
    assert parsed["output"] == "FM"
    assert parsed["region"] == "BAQ"
    assert parsed["metrics"]["mae"] == 0.1
    assert parsed["config_fingerprint"] == config_fingerprint(config)
    assert parsed["run_id"] == record["run_id"]
