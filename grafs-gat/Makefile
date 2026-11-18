# GRAFS-GAT: one-word entry points for every reproduction step (run from grafs-gat/).
#
#   make env       create the conda environment (exact package versions)
#   make check     environment + data check (compares patient/event counts with the paper)
#   make smoke     pipeline test on synthetic data (~2 min, no patient data needed)
#   make verify    download release v0.1.0-models (v3, earlier 199-patient analysis) and reproduce its AUCs
#                  (MAASTRO only; writes to outputs_verify_v3/ so the v4 study cache in outputs/ is untouched)
#   make study     full study from scratch (preprocessing v4 = manuscript; several GPU-days)
#   make export    collect results -> $GRAFS_OUTPUT_DIR/results/ (no retraining)
#   make paper     regenerate every figure, table and number of the manuscript
#   make docker    build the Docker image
#
# Override the interpreter with PY=..., e.g. make smoke PY=/opt/conda/envs/GNNvenv/bin/python

PY ?= python

.PHONY: env check smoke verify study export paper docker

env:
	conda env create -f environment.yml

check:
	$(PY) check_setup.py

smoke:
	$(PY) smoke_test.py

VERIFY_DIR ?= outputs_verify_v3

verify:
	GRAFS_OUTPUT_DIR=$(VERIFY_DIR) $(PY) verify_release.py --download

study:
	PY=$(PY) ./run_mateus_study.sh

export:
	$(PY) export_results.py

paper:
	$(PY) ../Paper/manuscript/scripts/make_paper_assets.py

docker:
	docker build -t grafs-gat .
