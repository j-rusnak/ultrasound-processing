# Contributing

## Development Setup

Install the package in editable mode:

```powershell
pip install -e ".[dev]"
```

Run tests with a repository-local temp directory:

```powershell
python -m pytest --basetemp .pytest_tmp
```

## Data Handling

Do not commit clinical images, label CSVs, checkpoints, predictions, Grad-CAM outputs, or generated reports. Keep real data under the gitignored `data/` and `artifacts/` directories.

## Experiment Changes

When adding an experiment:

- add a config under `configs/experiments/`;
- keep paths config-driven;
- include the target output (`FM` or `FFM`) and anatomical region code;
- record metrics with the existing evaluation scripts;
- add or update tests for any new reusable code.

## Pull Request Checklist

- `python -m pytest --basetemp .pytest_tmp`
- CLI dry-run for any new config
- README or docs updated for user-facing workflow changes
- no private data or generated artifacts in `git status`
