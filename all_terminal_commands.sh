set -e
export PYTHONPATH=src:scripts
python scripts/check_data.py --raw data/raw
python scripts/build_manifest.py
python scripts/build_panel.py
python scripts/make_scenarios.py
python scripts/run_experiments.py
python scripts/run_model_selection.py
python scripts/run_bonfrank.py
python scripts/run_ablation.py
python scripts/run_margins.py
python scripts/run_trend.py
python scripts/run_final_panel.py
python scripts/run_scalar_baselines.py
python scripts/run_multiname.py
python scripts/run_multiname_margin.py
python scripts/make_tables.py
python scripts/make_figures.py
