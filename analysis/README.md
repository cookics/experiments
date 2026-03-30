# Analysis Code

This package contains the analysis, plotting, and model-training scripts that were previously loose at the repository root.

## Layout

- `analysis/` - shared helpers and runnable analysis modules
- `submit.py` - submission summarizer for experiment folders
- `update_score.py` - utility for recalculating summary statistics

## Running Scripts

From the repository root, you can run either of these forms:

```bash
python analysis/analyze_human_dataset_effort.py
python -m analysis.analyze_human_dataset_effort
```

Most scripts write derived artifacts into `analysis_outputs/`, which stays out of Git.

## Data Sources

The code expects local human-run archives, NetHack logs, and submission exports to be present on disk, but those directories are ignored by Git and are not part of the repository sync.
