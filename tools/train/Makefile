# SPDX-License-Identifier: MIT
#
# tools/train/Makefile — Training pipeline for 1bit.systems
#
# Integrates with the rest of 1bit's build system.
# Training runs in Docker; the Rust binary never touches Python.
#
# Usage:
#   make docker           # Build the training container
#   make train-lora       # Train Qwen3-0.6B LoRA on Alpaca
#   make train-rl         # GRPO on GSM8K
#   make export           # Export checkpoint to GGUF
#   make train            # Full pipeline: train → export
#   make clean            # Remove outputs and images

DOCKER_IMAGE := 1bit-train
CONFIGS_DIR := tools/train/configs
OUTPUTS_DIR := outputs
EXPORT_DIR  := export

# Default target
.PHONY: default
default: docker

# ── Container ─────────────────────────────────────────────────────────────────

.PHONY: docker
docker:
	docker build -f tools/train/Dockerfile -t $(DOCKER_IMAGE) tools/train/

# ── Training ──────────────────────────────────────────────────────────────────

.PHONY: train-sft
train-sft: docker
	mkdir -p $(OUTPUTS_DIR)
	docker run --rm --gpus all \
		-v $(PWD)/$(OUTPUTS_DIR):/outputs \
		-v $(PWD)/$(CONFIGS_DIR):/configs:ro \
		$(DOCKER_IMAGE) \
		/train_sft.py --config /configs/qwen3-0.6b-lora.yaml --output /outputs

.PHONY: train-rl
train-rl: docker
	mkdir -p $(OUTPUTS_DIR)
	docker run --rm --gpus all \
		-v $(PWD)/$(OUTPUTS_DIR):/outputs \
		-v $(PWD)/$(CONFIGS_DIR):/configs:ro \
		$(DOCKER_IMAGE) \
		/train_rl.py --config /configs/qwen3-0.6b-grpo.yaml --output /outputs

.PHONY: train-full-ft
train-full-ft: docker
	mkdir -p $(OUTPUTS_DIR)
	docker run --rm --gpus all \
		-v $(PWD)/$(OUTPUTS_DIR):/outputs \
		-v $(PWD)/$(CONFIGS_DIR):/configs:ro \
		$(DOCKER_IMAGE) \
		/train_sft.py --config /configs/qwen3-0.6b-full-ft.yaml --output /outputs

.PHONY: train-lora-zaya
train-lora-zaya: docker
	mkdir -p $(OUTPUTS_DIR)
	docker run --rm --gpus all \
		-v $(PWD)/$(OUTPUTS_DIR):/outputs \
		-v $(PWD)/$(CONFIGS_DIR):/configs:ro \
		$(DOCKER_IMAGE) \
		/train_sft.py --config /configs/zaya-8b-lora.yaml --output /outputs

# ── Export ────────────────────────────────────────────────────────────────────

# Find latest checkpoint and export to GGUF
.PHONY: export
export: docker
	@latest=$$(ls -td $(OUTPUTS_DIR)/*/checkpoint-* 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then \
		echo "No checkpoints found in $(OUTPUTS_DIR)/"; \
		exit 1; \
	fi; \
	echo "Exporting: $$latest"; \
	mkdir -p $(EXPORT_DIR); \
	docker run --rm --gpus all \
		-v $(PWD)/$(OUTPUTS_DIR):/outputs \
		-v $(PWD)/$(EXPORT_DIR):/export \
		$(DOCKER_IMAGE) \
		/export.py /outputs/$$(basename $$(dirname $$latest))/$$(basename $$latest) \
			--output-dir /export --format gguf

.PHONY: export-all
export-all: docker
	mkdir -p $(EXPORT_DIR)
	for ckpt in $$(find $(OUTPUTS_DIR) -type d -name "checkpoint-*" | sort); do \
		echo "Exporting: $$ckpt"; \
		rel=$$(echo $$ckpt | sed 's|$(OUTPUTS_DIR)/||'); \
		docker run --rm --gpus all \
			-v $(PWD)/$(OUTPUTS_DIR):/outputs \
			-v $(PWD)/$(EXPORT_DIR):/export \
			$(DOCKER_IMAGE) \
			/export.py /outputs/$$rel --output-dir /export --format gguf; \
	done

# ── Full pipeline ─────────────────────────────────────────────────────────────

.PHONY: train
train: train-sft export

.PHONY: train-rl-full
train-rl-full: train-rl export

# ── Interactive (debug) ───────────────────────────────────────────────────────

.PHONY: shell
shell: docker
	docker run --rm -it --gpus all \
		-v $(PWD)/$(OUTPUTS_DIR):/outputs \
		-v $(PWD)/$(CONFIGS_DIR):/configs:ro \
		--entrypoint /bin/bash \
		$(DOCKER_IMAGE)

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf $(OUTPUTS_DIR) $(EXPORT_DIR)

.PHONY: clean-docker
clean-docker:
	docker rmi $(DOCKER_IMAGE) 2>/dev/null || true

.PHONY: clean-all
clean-all: clean clean-docker
