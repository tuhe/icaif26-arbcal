# ArbCal: arbitrage-free conformal prediction sets and scenarios for option surfaces

Anonymized code release for an ICAIF 2026 submission.

## Layout

- `src/arbcal/` — the core package: no-arbitrage polyhedron and
  projection, surface panel construction, adaptive conformal
  calibration, joint spot-surface scenario generation, book repricing
- `scripts/` — pipeline stages, run in the order given below
- `manifests/` — the trading-date manifest and the raw-data audit
  manifest (expected files, row counts, sha256)
- `results/`, `figures/` — created by the scripts (not committed)

## Install (about 5 minutes)

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
export PYTHONPATH=src:scripts
```

Python 3.11 or newer.

## Data

No market data is redistributed in this repository. The daily study
uses SPX end-of-day option chains from OptionsDX (the monthly
`spx_eod_*.7z` archives from 2017 through 2023). Purchase or download
the archives and unpack the monthly `.txt` files into `data/raw/`.
The weekly multi-underlying study uses OptionMetrics IvyDB via WRDS
(options, security price, and security name tables for the 16
underlyings listed in the paper); export the pulls to
`data/raw/ivydb_multiname/` as parquet.

## Verify the data

```bash
python scripts/check_data.py --raw data/raw
```

This checks the unpacked OptionsDX files against
`manifests/optionsdx_manifest.json` (presence, row counts, sha256) so
that any deviation from the paper's inputs is caught before running
anything.

## Run

```bash
sh all_terminal_commands.sh          # full pipeline (hours; see notes)
```

The stages, in order:

1. `build_manifest.py` — trading-calendar manifest from the raw archives
2. `build_panel.py` — arbitrage-free surface panel (one 21x5 grid per session)
3. `make_scenarios.py` — joint spot-surface scenario chunks (slow stage)
4. `run_experiments.py` — surface sets and loss limits, all main arms
5. `run_model_selection.py`, `run_bonfrank.py`, `run_ablation.py`,
   `run_margins.py`, `run_trend.py`, `run_final_panel.py`,
   `run_scalar_baselines.py` — remaining study arms
6. `run_multiname.py`, `run_multiname_margin.py` — weekly 16-name study
7. `make_tables.py` — prints the paper's tables to the terminal
8. `make_figures.py` — writes the online-adaptation figure as a PNG

Approximate cost on a consumer CPU: the scenario stage is the long one
(projection of 10,000 draws is roughly 10 seconds per date across
1,657 dates); the remaining stages take minutes to tens of minutes.

Status note: this release accompanies the submission; a validation
pass regenerating every table from a clean checkout is in progress and
the repository will be updated in place.
