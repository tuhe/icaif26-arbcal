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

Python 3.11 or newer. Install the full `requirements.txt` as given: every
entry is load-bearing, and `scikit-learn` in particular is not optional —
without it the scenario covariance silently falls back to a different
shrinkage estimator, and the pipeline reproduces cleanly but with different
numbers. `numpy`, `scipy` and `osqp` are pinned so that the panel and the
scenario draws come out bit-identical; see the comment in the file if you
want to relax them.
`build_manifest.py` additionally shells out to the `7z` binary (`p7zip-full`)
to read archive names; it is the only stage that does.

## Data

No market data is redistributed in this repository. The daily study
uses SPX end-of-day option chains from OptionsDX (the monthly
`spx_eod_*.7z` archives from 2017 through 2023). Purchase or download
the archives and unpack the monthly `.txt` files into `data/raw/`.
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

The stages, in order:

1. `build_panel.py` — arbitrage-free surface panel from the raw chains
   (one 21x5 grid per session; ~2 minutes for all 84 months)
2. `build_manifest.py` — trading-calendar manifest and cleaning audit.
   Optional: it regenerates the committed `manifests/date_manifest.csv`,
   and needs the original `.7z` archives for the provenance column.
3. `filter_sessions.py` — restrict the panel to the manifest's retained
   sessions and assert the day list matches the committed manifest
   exactly (1,739 sessions). This is the anchor: if your rebuilt panel
   disagrees with the paper's, you find out here rather than in a table.
4. `make_scenarios.py` — joint spot-surface scenario chunks. Two phases:
   `--phase a` (1,000 surface draws, feeds stage 5) and `--phase b`
   (10,000 loss draws, feeds stage 6). Phase b is needed once per
   scenario law: `--law gauss` and `--law t --nu {3,5,8}`. This is the
   slow stage; see the sharding note below.
5. `run_experiments.py` — surface sets and loss limits, all main arms
   (consumes the phase-a chunks; writes `fullcal_canonical.json` and
   `fullcal_arrays.npz`, which later stages read)
6. `run_model_selection.py` (chooses among gauss/t3/t5/t8; the paper
   selects t3), `run_bonfrank.py`, `run_ablation.py`, `run_limits_m.py`,
   `run_margins.py`, `run_trend.py`, `run_final_panel.py`,
   `run_scalar_baselines.py` — remaining study arms
7. `run_multiname.py`, `run_multiname_margin.py` — weekly 16-name study
   (needs the WRDS/IvyDB pulls, not the OptionsDX archives)
8. `make_main_tables.py`, `make_ablation_table.py` — write the LaTeX tables
   used in the paper into `results/`
9. `make_tables.py` — prints those results to the terminal as plain tables
10. `make_figures.py` — the online-adaptation figure

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
generators write `.tex` into `results/`. Figure 1 in the paper is a
hand-drawn schematic, not a result, and has no generator here.

## Reproduction status

Verified on 2026-08-10 by running this repository from a clean checkout
and a clean virtual environment against the OptionsDX archives:

- `check_data.py` — 84/84 months sha256-identical to the manifest.
- `build_panel.py` + `filter_sessions.py` — the rebuilt panel is
  **bit-identical** to the one behind the paper (spot, forwards,
  discounts and all 21x5 surfaces), and lands on exactly the 1,739
  retained sessions.
- `make_scenarios.py --phase a` — chunks **bit-identical** to the
  paper's, verified under two different numpy/scipy/Python combinations.
- `run_experiments.py` — `fullcal_canonical.json` matches the paper's,
  including the adaptive-conformal learning rates selected on the
  development period; the largest deviation across 1,730 reported
  numeric values is 1.4e-08 (JSON round-trip precision).
- `run_model_selection.py` — independently selects the same scenario law
  the paper uses (Student-t, 3 degrees of freedom) from the
  {gauss, t3, t5, t8} pool.
- every other daily-study stage — the six result JSONs agree to at worst
  1.4e-07, and the ablation file is bit-exact.
- **all 18 generated LaTeX tables are byte-identical** to the ones in the
  paper, including the two wired into the manuscript body
  (`tab_limits_main_t3_2018_2022`, `tab_ablation_main_t3_2018_2022`). The
  headline held-out-2023 reserve pair reproduces exactly: 1.8829 for
  ArbCal-Rank against 3.3817 for the per-book static Bonferroni baseline.
- `run_multiname.py` / `run_multiname_margin.py` — the weekly
  16-underlying study runs end to end from the IvyDB pull, reproduces the
  paper's mean coverage figures at the printed precision, and its margin
  file is bit-exact across all 71 reported values.
- `make_figures.py` — the online-adaptation figure comes out
  **pixel-identical** to the one in the paper (compared after rasterising at
  equal resolution; only the timestamp matplotlib embeds differs).

**Everything the paper reports reproduces from this repository.** Every
stage runs, every table in the body regenerates byte-identically, and the
reported numbers — including the headline held-out-2023 reserve pair and
the weekly verification study — come out of the pipeline published here,
given the raw archives described above.
