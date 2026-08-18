# Arbitrage-free conformal prediction sets and scenarios for option surfaces

Anonymized code for a full replication of an ICAIF 2026 submission.


## Content

- `src/arbcal/`: The core package containing no-arbitrage polyhedron and
  projection, surface panel construction, adaptive conformal
  calibration, joint spot-surface scenario generation, book repricing
- `scripts/`: The pipeline stages, run in the order given below
- `manifests/`: the trading-date manifest and the raw-data audit
  manifest (expected files, row counts, sha256), used to check the data is downloaded correctly
- `results/`, `figures/`: created by running the above scripts and contains the results.

## Install
Simply run:


```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
export PYTHONPATH=src:scripts
```

Use Python 3.11 or newer.
`build_manifest.py` additionally prints out the `7z` binary (`p7zip-full`)
used to read archive names.

## Data

No market data is redistributed in this repository. The daily study
uses SPX end-of-day option chains from OptionsDX (the monthly
`spx_eod_*.7z` archives from 2017 through 2023). This data can be downloaded for free from OptionsDX. To do so:

 - Go to https://www.optionsdx.com/?post_type=product
 - Select "SPX Options Chains" (second entry)
 - Add all relevant years to the cart
 - Go to checkout and download (the options are free)

Next, unpack the monthly `.txt` files into `data/raw/`.
The weekly multi-underlying study uses OptionMetrics IvyDB via WRDS
(options, security price, and security name tables for the 16
underlyings listed in the paper), expected at
`data/raw/ivydb_multiname/` as parquet. That pull is scripted, so a
licensed user can rebuild it rather than reconstruct the query:

```bash
pip install wrds                    # only needed for this step
export WRDS_USERNAME=<your-wrds-user>
python scripts/pull_ivydb.py        # writes data/raw/ivydb_multiname/
```

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

The stages to run, in order:

1. `build_panel.py`: arbitrage-free surface panel from the raw chains
2. `build_manifest.py`: trading-calendar manifest and cleaning audit.
   Optional: it regenerates the committed `manifests/date_manifest.csv`,
   and needs the original `.7z` archives for the provenance column.
3. `filter_sessions.py`: restrict the panel to the manifest's retained
   sessions and assert the day list matches the committed manifest
   exactly (1,739 sessions). If your rebuilt panel
   disagrees with the paper's, you will find out in this step
4. `make_scenarios.py`: joint spot-surface scenario chunks. Two phases:
   `--phase a` (1,000 surface draws, feeds stage 5) and `--phase b`
   (10,000 loss draws, feeds stage 6). Phase b is needed once per
   scenario law: `--law gauss` and `--law t --nu {3,5,8}`. This is the
   slow stage,  see the sharding note below.
5. `run_experiments.py`: surface sets and loss limits, all main experiments
   (consumes the phase-a chunks; writes `fullcal_canonical.json` and
   `fullcal_arrays.npz`, which later stages read)
6. `run_model_selection.py` (chooses among gauss/t3/t5/t8; the paper
   selects t3), `run_bonfrank.py`, `run_ablation.py`, `run_limits_m.py`,
   `run_margins.py`, `run_trend.py`, `run_final_panel.py`,
   `run_scalar_baselines.py`, which is the remaining study experiment
7. `run_multiname.py`, `run_multiname_margin.py`: weekly 16-name study
   (needs the WRDS/IvyDB pulls in addition to the OptionsDX archives)
8. `make_main_tables.py`, `make_ablation_table.py`: write the LaTeX tables
   used in the paper into `results/`
9. `make_tables.py`: prints those results to the terminal as plain tables

Approximate cost on a consumer CPU: the scenario stage dominates
(projection of 10,000 draws is roughly 10 seconds per date across 1,657
dates, per law, so about 4 hours each single-process); everything else
takes minutes. `make_scenarios.py` accepts `--t-start/--t-end`, and the
per-date seeding is `default_rng(SEED_BASE + t)`, so splitting the date
range across processes is bit-identical to one long run and brings the
scenario stage to well under an hour on a many-core machine.

## Tables and figures on their own

`all_terminal_commands.sh` already runs these at the end, so a full run
needs nothing extra. To redo just the reporting from existing `results/`:

```bash
python scripts/make_main_tables.py                      # LaTeX -> results/
python scripts/make_ablation_table.py --panel data/panel_sessions.npz \
    --losses-glob 'data/scen/scen_b_t3_*.npz' --nu 3
python scripts/make_tables.py --results results         # print to terminal
python scripts/make_figures.py --panel data/panel_sessions.npz \
    --losses-glob 'data/scen/scen_b_t3_*.npz' \
    --canonical results/fullcal_canonical.json --out figures/fig_online_t3_95.pdf
```

`make_tables.py` reads the result JSONs and prints them with `tabulate`, so
you can read the headline numbers without opening a LaTeX file; the other
generators write `.tex` into `results/`.

