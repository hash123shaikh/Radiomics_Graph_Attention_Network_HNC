# RadGraph-FS: reproducible environment (Python 3.8, PyTorch 2.0.1 + CUDA 11.7, PyG 2.3.1, PyRadiomics 3.0.1).
#
#   docker build -t radgraph-fs .                                   (from radgraph-hnscc/)
#   docker run --rm radgraph-fs                                     (smoke test on synthetic data)
#   docker run --rm --gpus all -v /path/to/data:/data -v /path/to/outputs:/outputs radgraph-fs \
#          python verify_release.py --download                      (reproduce the published external AUCs)
#   docker run --rm --gpus all -v /path/to/data:/data -v /path/to/outputs:/outputs radgraph-fs \
#          ./run_mateus_study.sh                                    (full study)
#
# /data must contain HNPET/ (or "HECKTOR 2026 Training Data"/) and MAASTRO/ as described in DATA.md.
# Runs on CPU when no GPU is passed (--gpus all needs the NVIDIA Container Toolkit).
FROM python:3.8-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir numpy==1.24.3 "setuptools>=65" \
    && pip install --no-cache-dir -r requirements-lock.txt

COPY . .

ENV RADGRAPH_DATA_DIR=/data \
    RADGRAPH_OUTPUT_DIR=/outputs \
    RADGRAPH_LOGS_DIR=/outputs/logs \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

CMD ["python", "smoke_test.py"]
