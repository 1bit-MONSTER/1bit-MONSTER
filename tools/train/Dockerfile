# SPDX-License-Identifier: MIT
#
# tools/train/Dockerfile — Training container for 1bit.systems
#
# Build:   docker build -f tools/train/Dockerfile -t 1bit-train tools/train/
# Run:     docker run --rm --device /dev/kfd --device /dev/dri \
#            -v $(pwd)/outputs:/outputs 1bit-train \
#            /train_sft.py --config /configs/qwen3-0.6b-lora.yaml --output /outputs
#
# Produces GGUF files that 1bit's Rust binary consumes directly.
# Zero Python ships to end users — training is CI/build-time only.

FROM rocm/dev-ubuntu-24.04:latest

# Python 3.12 is already in the base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv \
    curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Virtual env for isolated Python
RUN uv venv /venv --python 3.12
ENV PATH="/venv/bin:${PATH}"
ENV VIRTUAL_ENV=/venv

# ROCm PyTorch from dedicated index (extra-index so deps come from PyPI)
RUN uv pip install --no-cache \
    --extra-index-url https://download.pytorch.org/whl/rocm7.2 \
    torch torchvision --force-reinstall

# Unsloth (Apache 2.0 — no unsloth_cli, no studio)
RUN uv pip install --no-cache \
    --extra-index-url https://download.pytorch.org/whl/rocm7.2 \
    unsloth

# Training scripts
COPY train_sft.py /train_sft.py
COPY train_rl.py  /train_rl.py
COPY export.py    /export.py
COPY configs/     /configs/

ENTRYPOINT ["python3"]
