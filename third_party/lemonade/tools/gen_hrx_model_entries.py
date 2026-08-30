#!/usr/bin/env python3
"""gen_hrx_model_entries.py — add `*-HRX` variants of the llamacpp models.

Goal (docs/research/hrx-engine-goal.md, P3): "all lemonade models also with
HRX". The `llamacpp-hrx` recipe (HRX llama-server, chat-only local-GGUF) can
serve any GGUF checkpoint the `llamacpp` recipe can; the registry just needs
entries. This script adds a `<Name>-HRX` entry for every `llamacpp` model that
qualifies for HRX:

- excluded: embeddings/reranking models (HRX is chat-only) and vision-labeled
  models (HRX's chat branch omits mmproj — unverified).
- naming: trailing "-GGUF" → "-HRX" (upstream pattern, e.g.
  Qwen3-30B-A3B-Instruct-2507-GGUF → ...-HRX), else append "-HRX"; entries
  that already exist are skipped (the shipped Qwen3-30B-A3B-Instruct-2507-HRX
  survives untouched).
- fields: same checkpoint, recipe "llamacpp-hrx", same labels/size,
  suggested=false (variants; the base GGUF keeps its suggested slot so the UI
  isn't flooded with 76 "suggested" cards).

KNOWN LIMITATION (2026-08-29): lemond has no failover, and the HRX runtime
fail-closes at GET_ROWS for non-K-quant token embeddings (q5_0/q8_0/IQ2XXS/
Q4_K_S verified). Generated entries therefore serve K-quant-embedding GGUFs
only; for other models users should pick the `llamacpp` (Vulkan/HIP) variant
or the engine-native path (1bit unified, which failovers).

Usage: python3 gen_hrx_model_entries.py [--write]
       (default: dry-run, prints the plan)
"""
import json
import sys
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "src" / "cpp" / "resources" / "server_models.json"
VERSIONS = Path(__file__).resolve().parent.parent / "src" / "cpp" / "resources" / "backend_versions.json"

EXCLUDE_LABELS = {"embeddings", "reranking", "vision"}


def hrx_name(name: str) -> str:
    return name[:-len("-GGUF")] + "-HRX" if name.endswith("-GGUF") else name + "-HRX"


def main() -> int:
    registry = json.loads(REGISTRY.read_text())
    versions = json.loads(VERSIONS.read_text())
    assert "llamacpp-hrx" in versions, "backend_versions.json missing llamacpp-hrx pin"

    new_entries, skipped = {}, {}
    for name, entry in registry.items():
        if not isinstance(entry, dict) or entry.get("recipe") != "llamacpp":
            continue
        labels = set(entry.get("labels", []))
        if labels & EXCLUDE_LABELS:
            skipped[name] = f"excluded: labels {sorted(labels & EXCLUDE_LABELS)}"
            continue
        target = hrx_name(name)
        if target in registry:
            skipped[name] = "already has an -HRX entry"
            continue
        new_entries[target] = {
            "checkpoint": entry["checkpoint"],
            "recipe": "llamacpp-hrx",
            "suggested": False,
            "labels": entry.get("labels", []),
            "size": entry.get("size", 0),
        }

    print(f"llamacpp models: {sum(1 for v in registry.values() if isinstance(v, dict) and v.get('recipe')=='llamacpp')}")
    print(f"qualifying -> new -HRX entries: {len(new_entries)}")
    for name in sorted(new_entries)[:10]:
        print(f"  + {name}")
    print(f"  ... ({len(new_entries) - min(10, len(new_entries))} more)")
    for name, why in sorted(skipped.items())[:8]:
        print(f"  - {name}: {why}")
    print(f"skipped: {len(skipped)}")

    if "--write" in sys.argv:
        registry.update(new_entries)
        REGISTRY.write_text(json.dumps(registry, indent=4, ensure_ascii=False) + "\n")
        print(f"WROTE {len(new_entries)} entries to {REGISTRY}")
        json.loads(REGISTRY.read_text())  # parse check
        print("parse check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
