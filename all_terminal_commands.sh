#!/bin/sh
# Full daily-study pipeline, single process, in dependency order.
#
# Raw OptionsDX monthly .txt files are expected in data/raw (see README).
# Every stage is restartable: outputs land in data/ and results/, and a
# stage only reads what earlier stages wrote.
#
# The scenario stage is the slow one (~4 hours per law single-process).
# It takes --t-start/--t-end and seeds per date, so it can be split across
# processes bit-identically; see the sharding note in the README.
set -e
export PYTHONPATH=src:scripts

PANEL=data/panel_sessions.npz
SCEN=data/scen
mkdir -p "$SCEN" results figures

# --- inputs -----------------------------------------------------------------
python scripts/check_data.py --raw data/raw

# --- panel ------------------------------------------------------------------
python scripts/build_panel.py --data-dir data/raw \
    --out data/panel_full.npz --diag results/panel_diag.json

# Optional: regenerates the committed manifests/date_manifest.csv. Needs the
# original .7z archives (--archive-dir) for the archive-provenance column.
# python scripts/build_manifest.py --data-dir data/raw --archive-dir data/archives \
#     --panel data/panel_full.npz --outdir results

python scripts/filter_sessions.py --panel data/panel_full.npz --out "$PANEL"

# --- scenarios --------------------------------------------------------------
# phase a: 1,000 surface draws, feeds run_experiments / run_margins
python scripts/make_scenarios.py --panel "$PANEL" --phase a \
    --out "$SCEN/scen_a_all.npz"

# phase b: 10,000 loss draws per candidate scenario law. The development
# period selects among these; the paper selects t3.
python scripts/make_scenarios.py --panel "$PANEL" --phase b --law gauss \
    --out "$SCEN/scen_b_gauss_all.npz"
for NU in 3 5 8; do
    python scripts/make_scenarios.py --panel "$PANEL" --phase b --law t --nu "$NU" \
        --out "$SCEN/scen_b_t${NU}_all.npz"
done

# --- main arms --------------------------------------------------------------
python scripts/run_experiments.py --panel "$PANEL" \
    --scen-glob "$SCEN/scen_a_*.npz" --outdir results

CANON=results/fullcal_canonical.json
GAUSS="$SCEN/scen_b_gauss_*.npz"     # phase-b Gaussian losses = the main arm
SEL="$SCEN/scen_b_t3_*.npz"          # phase-b losses under the selected law

# Which glob goes where is easy to get wrong, and wrong pairings still run to
# completion with plausible-looking numbers:
#   --scen-glob    phase-a chunks (surface draws, 1,000)
#   --losses-glob  phase-b chunks (loss draws, 10,000) under the arm's own law
#   --t-glob       phase-b chunks under the *selected* Student-t law
# Stages carrying a --law/--tag/--t-law-name flag default to gauss, and the
# flag must match the chunks actually passed.

python scripts/run_model_selection.py --panel "$PANEL" \
    --gauss-glob "$GAUSS" \
    --t-globs "{\"3\":\"$SCEN/scen_b_t3_*.npz\",\"5\":\"$SCEN/scen_b_t5_*.npz\",\"8\":\"$SCEN/scen_b_t8_*.npz\"}" \
    --canonical "$CANON" --out results/fullcal_tsel.json

# Writes fullcal_canonical_b.json AND fullcal_canonical_b_arrays.npz (the
# arrays name is derived from --out) — later stages read both, so do not
# rename the output. Gaussian losses are the main arm; the selected t-law
# enters as a second arm via --t-glob, labelled by --t-law-name.
python scripts/run_limits_m.py --panel "$PANEL" --losses-glob "$GAUSS" \
    --t-glob "$SEL" --t-law-name t3 \
    --canonical "$CANON" --out results/fullcal_canonical_b.json

python scripts/run_bonfrank.py --panel "$PANEL" \
    --scen-glob "$SCEN/scen_a_*.npz" --outdir results

python scripts/run_ablation.py --panel "$PANEL" --losses-glob "$GAUSS" \
    --canonical "$CANON" --out results/fullcal_abl10k.json

python scripts/run_margins.py --panel "$PANEL" --outdir results

python scripts/run_trend.py --panel "$PANEL" --out results/fullcal_trend_diag.json

# Once per law; --tag selects the output names and the cross-check anchor.
# make_main_tables.py reads these from results/q2q3_final_panel/, so keep the
# subdirectory (run_scalar_baselines.py likewise writes q2q3_soft_baselines/).
python scripts/run_final_panel.py --panel "$PANEL" --losses-glob "$GAUSS" \
    --tag gauss --outdir results/q2q3_final_panel
python scripts/run_final_panel.py --panel "$PANEL" --losses-glob "$SEL" \
    --tag t3 --outdir results/q2q3_final_panel

python scripts/run_scalar_baselines.py

# --- weekly 16-name study ---------------------------------------------------
# Needs the IvyDB pull (scripts/pull_ivydb.py) rather than the OptionsDX
# archives; skip these two if you only have the daily inputs.
python scripts/run_multiname.py --out results/vs_multiname_joint.json
python scripts/run_multiname_margin.py --out results/vs_multiname_margin.json

# --- reporting --------------------------------------------------------------
# the tables wired into the paper body
python scripts/make_main_tables.py
python scripts/make_ablation_table.py --panel "$PANEL" --losses-glob "$SEL" --nu 3
# terminal summary of the headline tables
python scripts/make_tables.py --results results
# figure
python scripts/make_figures.py --panel "$PANEL" --losses-glob "$SEL" \
    --canonical "$CANON" --out figures/fig_online_t3_95.pdf
