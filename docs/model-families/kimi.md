# Moonshot Kimi — Gated MLA MoE

Moonshot AI's Kimi family (Moonlight, Kimi-VL) uses a Gated MLA MoE architecture. When a major model drops, 1bit doesn't wait for vendor support — Kimi's architecture was reverse-engineered from the open weights and a 1BP converter was built.

## Models

| Model | Params | 1BP Size | Backend(s) | Status |
|-------|:------:|:--------:|------------|:------:|
| **Kimi (Moonlight)** | 16B (3B active) | — | GPU HIP | 🔄 arch analyzed, converter built |
| **Kimi-VL** | — | — | GPU HIP (vision) | 🔄 arch analyzed |

## Notes

- Architecture reverse-engineered (Gated MLA MoE); converter built. Integration in progress.
- Same pure-C++ reverse-engineering process used for the NPU stack — no Python involved. See the [reverse-engineering notes](../research/kimi-k3-reverse-engineering.md).

**See also:** [engineering journey](../journey.md) · [full model support detail](../wiki/models.md) · [all families](README.md)
