# RadGraph-FS: one-word entry points for every reproduction step (run from radgraph-hnscc/).
#
#   make env       create the conda environment (exact package versions)
#   make check     environment + data check (compares patient/event counts with the paper)
#   make smoke     pipeline test on synthetic data (~2 min, no patient data needed)
#   make verify    download the released models and reproduce the published external AUCs (MAASTRO only)
#   make study     full study from scratch (preprocessing v3 = manuscript; several GPU-days)
#   make export    collect results -> $RADGRAPH_OUTPUT_DIR/results/ (no retraining)
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

verify:
	$(PY) verify_release.py --download

study:
	PY=$(PY) ./run_mateus_study.sh

export:
	$(PY) export_results.py

paper:
	$(PY) ../Paper/manuscript/scripts/make_paper_assets.py

docker:
	docker build -t radgraph-fs .
