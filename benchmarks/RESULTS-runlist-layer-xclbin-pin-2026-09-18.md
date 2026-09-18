# The runlist's layer.xclbin is now pinned in the ENGINE, not just in the harnesses — 2026-09-18

Goal `mu35shsg-i3hlyi`. Fixes the class of defect behind the 2026-09-18 08:19 oracle
breakage (`RESULTS-oracle-scoreboard-corrected-2026-09-18.md`): the runlist bridge preferred
a **foreign, mutable build directory** over the in-repo pin, and validated it by
`stat()` existence only — never identity.

## The defect (as it was)

`engine/npu/src/npu_runlist_bridge.cpp`, both `npu_runlist_session_init` and
`npu_runlist_decode`:

```cpp
if (!getenv("LAYER_XCLBIN")) {
    const std::string base = "/home/bcloud/amd-oss/fastflowlm/src/xclbins/";
    std::string md = ...;                                  // model dir name
    if (md.empty() || stat((base + md + "/layer.xclbin").c_str(), &st) != 0)
        md = sess_model_dir(H);
    setenv("LAYER_XCLBIN", (base + md + "/layer.xclbin").c_str(), 0);
}
```

The repo carries the file the per-ctx ELFs were generated against
(`engine/npu/xclbins/flm_models/<Model>/layer.xclbin`, 339980 B, md5
`57431faab8593fadbffb5b9d5a9a0735` for Qwen3-0.6B) and the engine never consulted it.
When another lane rebuilt the amd-oss copy (401980 B, md5
`fa9f8df2f2b5618a560fd5470104aade`) the runlist silently emitted garbage: native 0/20 with
the FLM oracle still at 18/20 and I1 OK=20. Three engine builds from different days
reproduced it, so the resolved file — not any build — was the variable. This is a
guaranteed future failure mode, not an accident: **any fastflowlm rebuild silently changes
what the runlist arm executes.**

## The fix

`pinned_layer_xclbin_path()` + `resolve_layer_xclbin()`: prefer, in order,

1. `$NPU_XCLBIN_DIR/flm_models/<model>/layer.xclbin`
2. `engine/npu/xclbins/flm_models/<model>/layer.xclbin` (relative to CWD, as the rest of
   this file already resolves its paths)

and only then the foreign build directory — with a loud stderr warning naming the resolved
path and the reason. `LAYER_XCLBIN` set by the caller still wins. Both call sites now use
the helper, so the duplicate copy of the old logic is gone.

## Verification (all measured on strixhalo, `accel0` otherwise idle)

```
# (a) auto-resolved, LAYER_XCLBIN unset -- THE FIX GATE
env -u LAYER_XCLBIN NPU_RUNLIST=1 NPU_GREEDY=1 \
  ./engine/npu/build/npu_engine_qwen3_0_6b ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 12 /tmp/pp.txt
[runlist] LAYER_XCLBIN pinned to the in-repo copy: engine/npu/xclbins/flm_models/Qwen3-0.6B-NPU2/layer.xclbin
  [1] 151667  [2] 198  [3] 32313  [4] 11  [5] 279  [6] 1196  [7] 374  [8] 10161  [9] 911 ...
=== 10.0 ms/tok (100 tok/s) | tokens=12 ===        <- coherent, was garbage before
```

| check | result |
|---|---|
| 0.6B, `LAYER_XCLBIN` unset | pins; coherent stream (`151667 198 32313 11 …`) |
| 1.7B, `LAYER_XCLBIN` unset | pins; `151667 198 32313 11 279 1196 374 10161` |
| 1.7B, explicit `LAYER_XCLBIN` | **identical** stream — auto-resolution == explicit pin |
| fallback path (bogus `NPU_XCLBIN_DIR`, CWD outside the repo) | `WARNING: no in-repo pinned layer.xclbin … falling back to the foreign build directory` and the run proceeds |

The build (all model variants) completed with the change; the marker string is present in
every variant:

```
npu_engine_qwen3_0_6b 1   npu_engine_qwen3_1_7b 1   npu_engine_qwen3_4b 1   npu_engine_qwen3_8b 1
npu_engine_qwen3_vl_4b 1  npu_engine_llama 1        npu_engine_nanbeige4_1_3b 1  npu_engine_phi4_mini_4b 1
```

(Checked by `grep -a`, not by the build's exit code — `build_npu.sh`'s stale-object trap is
recorded in `LEVERS-register-2026-09-15.md` 6.3, and an interrupted build after the reboot
was re-run to completion before this check.)

## Scope and what it does NOT do

- It does **not** validate the pinned file's identity against the ELFs (no md5 is baked in).
  Preferring the in-repo pin plus the loud fallback warning removes the silent-swap failure
  mode; a strict identity gate would need the per-ctx ELFs to carry their generator's input
  hash, which is a bigger change.
- The harness-level `LAYER_XCLBIN` pin added in `87d16ed10` is now redundant but harmless —
  it still makes a run explicit.
- The 2026-09-16 `(c)` yardstick numbers were taken before the 08:19 replacement and are
  unaffected; any **re-run** of their gate arm now resolves the in-repo pin automatically,
  which is the point of doing this in the engine rather than per harness.

Cross-reference: `832dffbee` / `639444644` / `f15cc984d` (co-lane agent) record the same
breakage from the measurement side, including the three measurement conditions
(`LEVERS` §6.5) and the temp-index trap (§6.6).
