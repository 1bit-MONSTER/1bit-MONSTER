#!/usr/bin/env python3
"""
Download the benchmark models from their Hugging Face pointers.

Reads a benchmark config (default: benchmark_config_ci.json), and for every
selected model whose registry entry carries an `_hf` field (the Hugging Face
repo id the files come from), downloads the model's `gguf` / `mmproj` /
`mtp_draft` / component files into the exact local paths the config resolves
them to — so a subsequent `run_matrix.py --config <same config>` finds them.

Files already present are skipped (hf_hub_download's local_dir cache), so this
is cheap to re-run; on CI the model root lives in a persistent directory and
the download is a one-time cost per file.

Usage:
    python download_models.py [--config benchmark_config_ci.json] \
        [--models gemma4-12b,qwen36-35b-a3b]

Respects the same env overrides as the rest of the harness (BENCH_MODEL_ROOT,
BENCH_CONFIG, ...). Requires `pip install -U huggingface_hub`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _peek_config_arg(argv) -> str | None:
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return None


HERE = Path(__file__).resolve().parent
_cfg_arg = _peek_config_arg(sys.argv[1:]) or os.environ.get("BENCH_CONFIG") \
    or str(HERE / "benchmark_config_ci.json")
os.environ["BENCH_CONFIG"] = _cfg_arg

import config  # noqa: E402  (must come after BENCH_CONFIG is set)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, metavar="PATH",
                    help="benchmark config file (default: benchmark_config_ci.json)")
    ap.add_argument("--models", default=None,
                    help="comma list of model ids to fetch (default: the config's "
                         "defaults.models)")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    # The `_hf` repo pointers are documentation-only fields config.py does not
    # surface, so read them from the raw JSON alongside the resolved registry.
    raw_models = json.loads(Path(config.CONFIG_PATH).read_text(encoding="utf-8")).get("models", {})

    wanted = [m.strip() for m in (args.models or "").split(",") if m.strip()] \
        or list(config.DEFAULT_MODELS)
    failed = []
    for model_id in wanted:
        model = config.MODELS.get(model_id)
        repo = (raw_models.get(model_id) or {}).get("_hf")
        if model is None:
            print(f"[{model_id}] unknown model id (known: {', '.join(config.MODELS)})")
            failed.append(model_id)
            continue
        if not repo:
            print(f"[{model_id}] no `_hf` repo pointer in {config.CONFIG_PATH}; "
                  f"expecting the files to already exist locally")
            continue
        targets = [model.gguf, model.mmproj, model.mtp_draft,
                   *(model.components or {}).values()]
        for target in targets:
            if target is None:
                continue
            target = Path(target)
            if target.exists():
                print(f"[{model_id}] present: {target}")
                continue
            print(f"[{model_id}] downloading {repo} :: {target.name} -> {target.parent}")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(repo_id=repo, filename=target.name,
                                local_dir=str(target.parent))
            except Exception as ex:
                print(f"[{model_id}] FAILED {repo} :: {target.name}: {ex}")
                failed.append(f"{model_id}:{target.name}")
    if failed:
        raise SystemExit(f"model download(s) failed: {', '.join(failed)}")
    print("all requested model files are in place")


if __name__ == "__main__":
    main()
