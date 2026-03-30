# Data Processing Notes

This repository keeps the generated data local and out of Git. The code in `analysis/` reads those files, transforms them into smaller derived tables and plots, and writes the results into `analysis_outputs/`.

## Ignored Inputs

- `_balrog_src/` and `_nle_src/` are local source checkouts used as reference data.
- `nld-nao/` contains the raw NetHack log archives and extracted game traces.
- `submissions/` contains evaluation exports and run logs.
- Top-level archives, CSVs, SQLite databases, PNGs, PDFs, and similar artifact files are ignored by `.gitignore`.

## Processing Flow

1. Read the raw logs or submission exports from the local data directories.
2. Parse and normalize game trajectories into smaller intermediate tables.
3. Build summary CSVs, SQLite caches, and plots under `analysis_outputs/`.
4. Keep the raw archives and large binaries untracked so only the code is synced.

## Practical Result

The repository stays lightweight in Git while still preserving the scripts needed to reproduce the figures and summary tables from the local data.
