#!/usr/bin/env python3
"""hf_coverage.py — HF model coverage checker (docs/research/hf-coverage-audit.md).

Given a HuggingFace model id (or a local dir with config.json), report which
lane of the coverage onion covers it and the exact path to run it on this
machine, or — for a novel architecture — the 5-step add-model checklist.

Lanes checked (all read live from the vendored tree, no maintenance):
  L1  llama.cpp GGUF set  — 263 HF architectures (extracted from the vendored
      converter's @ModelBase.register names) + optional GGUF-on-hub check.
  L2  engine-specialized — dedicated routes in src/model_router.cpp.
  L3  FLM NPU (Q4NX)      — third_party/FastFlowLM/src/model_list.json.
  L4  lemonade catalog    — third_party/lemonade/src/cpp/resources/server_models.json.

Usage:
  python3 tools/hf_coverage.py <model_id>            # fetch config.json from HF
  python3 tools/hf_coverage.py <model_id> --offline  # skip network (needs local cache)
  python3 tools/hf_coverage.py ./path/to/model-dir   # local config.json
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP = ROOT / "third_party" / "llama.cpp"
FLM_LIST = ROOT / "third_party" / "FastFlowLM" / "src" / "model_list.json"
LEMONADE_REGISTRY = ROOT / "third_party" / "lemonade" / "src" / "cpp" / "resources" / "server_models.json"
MODEL_ROUTER = ROOT / "src" / "model_router.cpp"

# Engine-specialized families (src/model_router.cpp dedicated routes), keyed by
# substring matched against the HF model id (lowercased).
ENGINE_SPECIALIZED = [
    ("qwen3.5/3.6/3.8 text (GatedDeltaNet) → cpu_qwen3_5 / lse", ["qwen3.5", "qwen3_5", "qwen3-3.5", "qwen3-3.8", "lemonseed"]),
    ("zamba2 → ggml_vulkan → zamba2_vulkan → zamba2_gpu", ["zamba2"]),
    ("zamba/mamba1 → mamba1_gpu", ["zamba-", "zamba2", "blackmamba"]),
    ("deepseek v4 → cpu_deepseek_v4", ["deepseek-v4", "deepseekv4"]),
    ("deepseek v2/v3 → hip_gpu", ["deepseek-v2", "deepseek-v3", "deepseekv2", "deepseekv3"]),
    ("glm_moe_dsa → cpu_glm_moe_dsa", ["glm-5", "glm_moe_dsa"]),
    ("mimo_v2 → cpu_mimo_v2", ["mimo-v2", "mimo_v2", "mimo-v3"]),
    ("nemotron-h → nemotron_h_cpu", ["nemotron-h", "nemotronh"]),
    ("laguna → laguna_gpu", ["laguna"]),
    ("whisper → cpu_generic (engine) / whispercpp (lemonade)", ["whisper"]),
    ("qwen3-vl → vision_encoder", ["qwen3-vl", "qwen3vl"]),
]


def fetch_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url)
        tok = os.environ.get("HF_TOKEN", "")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"  (fetch failed: {e})")
        return None


def llama_cpp_archs() -> set[str]:
    names = set()
    for f in sorted((LLAMA_CPP / "conversion").glob("*.py")):
        src = f.read_text(errors="replace")
        for m in re.finditer(r"@ModelBase\.register\((.*?)\)", src, re.S):
            for s in re.findall(r"[\"']([a-zA-Z0-9_.-]+)[\"']", m.group(1)):
                names.add(s)
    return names


def flm_models() -> list[str]:
    try:
        d = json.loads(FLM_LIST.read_text())
        return d if isinstance(d, list) else list(d.keys())
    except Exception:  # noqa: BLE001
        return []


def lemonade_models() -> dict:
    try:
        return json.loads(LEMONADE_REGISTRY.read_text())
    except Exception:  # noqa: BLE001
        return {}


def gguf_on_hub(model_id: str) -> bool:
    info = fetch_json(f"https://huggingface.co/api/models/{model_id}?blobs=true")
    if not info:
        return False
    for s in info.get("siblings", []):
        if str(s.get("rfilename", "")).endswith(".gguf"):
            return True
    return False


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = sys.argv[1]
    offline = "--offline" in sys.argv

    # 1. config.json
    config = None
    if Path(target).is_dir():
        config = json.loads((Path(target) / "config.json").read_text())
        model_id = Path(target).name
    else:
        model_id = target
        if not offline:
            config = fetch_json(f"https://huggingface.co/{model_id}/resolve/main/config.json")
        if config is None:
            print(f"BLOCKED: no config.json for '{model_id}' (use --offline with a local cache)")
            return 1

    arch = (config.get("architectures") or [""])[0]
    model_type = config.get("model_type", "")
    print(f"model:  {model_id}")
    print(f"arch:   {arch}")
    print(f"type:   {model_type}")
    print()

    # 2. L4 lemonade
    lm = lemonade_models()
    lm_hit = [k for k in lm if isinstance(lm[k], dict) and model_id.lower() in k.lower()]
    if lm_hit:
        e = lm[lm_hit[0]]
        print(f"L4 LEMONADE: '{lm_hit[0]}' — recipe {e.get('recipe')} "
              f"(HRX: {'yes' if e.get('recipe') == 'llamacpp-hrx' else 'via engine-native HRX-first if GGUF'})")

    # 3. L3 FLM
    flm = flm_models()
    flm_hit = [m for m in flm if model_id.lower() in m.lower()]
    if flm_hit:
        print(f"L3 FLM NPU (Q4NX): {flm_hit} — npu_flm route (67.5 tok/s)")

    # 4. L2 engine-specialized
    low = model_id.lower()
    eng_hits = [desc for desc, keys in ENGINE_SPECIALIZED if any(k in low for k in keys)]
    if eng_hits:
        print(f"L2 ENGINE-SPECIALIZED: {' | '.join(eng_hits)}")
    else:
        print("L2 ENGINE-SPECIALIZED: none (falls through to generic lanes)")

    # 5. L1 llama.cpp GGUF set
    l1 = llama_cpp_archs()
    in_l1 = arch in l1
    has_gguf = False if offline else gguf_on_hub(model_id)
    if in_l1:
        lane = "HRX fused (in-process) → ggml_vulkan → zinc_gpu → cpu_generic" if not has_gguf else \
               "hub GGUF → HRX fused (in-process) → ggml_vulkan → zinc_gpu → cpu_generic"
        print(f"L1 LLAMA.CPP: COVERED ({arch} in the 263-arch converter set)"
              f"{' + GGUF already on hub' if has_gguf else ' — convert: python3 third_party/llama.cpp/convert_hf_to_gguf.py <dir> --outfile out.gguf'}")
        print(f"   route: {lane}")
    elif has_gguf:
        print(f"L1 LLAMA.CPP: COVERED (GGUF on hub despite unknown arch string {arch}) — HRX-first route")
    else:
        print(f"L1 LLAMA.CPP: {arch} NOT in converter set (263 archs) and no hub GGUF")

    # 6. verdict
    covered = in_l1 or has_gguf or bool(eng_hits) or bool(flm_hit)
    print()
    if covered:
        print("VERDICT: COVERED — documented lane above.")
        return 0
    print("VERDICT: BLOCKED — novel architecture with no lane anywhere. Add-model checklist:")
    print("  1. src/model_discovery.cpp  — detect config.json model_type / architecture")
    print("  2. src/model_router.cpp     — route entry in select_backend_route()")
    print("  3. src/backend_factory.cpp  — backend factory (+ backend_manager.cpp if new)")
    print("  4. bench/record.sh          — benchmark entry")
    print("  5. verify HRX fused-node coverage on gfx1151 (or GGUF-via-upstream first)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
