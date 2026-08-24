# 1bit-LLM — Ternary LLM Decode Accelerator (RTL)

**Status:** wiring pass 1 — skeleton datapath, simulation-verified
**Target:** open-source 7-series flow (iverilog sim → yosys synth_xilinx → nextpnr-xilinx + prjxray)
**Portability note:** the Alveo U250 (the eventual box per journey UPDATE 32) is UltraScale+ and has
no open-source bitstream path, so the RTL is written in portable Verilog-2001 with **no vendor
primitives** — only inferred BRAMs. The same RTL can be dropped into a Vivado U250 project later;
the open-source flow keeps the 7-series (Artix/Kintex-7) honest for free.

---

## 1. What this is

A first, wired, *correct* slice of the "1-bit FPGA LLM box": the decode-time workhorse — a
**ternary (1.58-bit) GEMV engine**. BitNet-style LLMs quantize weights to {-1, 0, +1}, which turns
the matmul into **pure adds/subtracts of activations** — no multipliers, no DSPs.

Scope of this pass:

- On-chip weight cache (BRAM), one layer-tile's worth.
- Activation buffer + output buffer (BRAM).
- 4-lane ternary dot-product datapath with int32 accumulators.
- Integer scale/shift output stage (bit-exact vs. a Python reference).
- A tiny host interface (register bus) so the whole thing is driven and verified without an SoC.
- Sim verification: two randomized cases, bit-exact vs `tools/gen_golden.py`.

Deliberately **not** in this pass (see §8 roadmap): external DRAM weight streaming, AXI, multi-
layer sequencing, softmax, KV cache, systolic-array throughput, PCIe.

## 2. Module hierarchy and wiring

```
                              hif (host bus: 16-bit word addr + 32-bit data)
                                          │
                    ┌─────────────────────▼──────────────────────┐
                    │  hif_slave  — reg file + address decode    │
                    │   CTRL STATUS CFG_K CFG_N SCALE            │
                    └───┬───────────────┬───────────────┬────────┘
                 start_pulse,cfg        │ port A (host) │ port A (host)
                 scale_q15,shift        │ 32-bit wr/rd  │ 32-bit wr/rd
                 busy/done/err  ┌───────▼───┐    ┌───────▼───┐
                    ┌───────────▼─┐        │ wmem │    │ xbuf  │
                    │  ctrl_fsm   │        │      │    │       │
                    │  sequencer  │        └───┬───┘    └───┬───┘
                    └─────┬───────┘            │ port B    │ port B
                          │                    │ (gemv)    │ (gemv)
        mac_en, clr_acc   │      w_in[7:0] ◄───┘  xb_rdata  x_addr
        act_in[7:0] ◄─────┼─────────────┘        b_addr ────┘
                          ▼
                 ┌───────────────────┐
                 │    gemv_core      │   4-lane ternary MAC tree,
                 │  acc[127:0] (4×32)│   int32 accumulators
                 └────────┬──────────┘
                          │
                 ┌────────▼──────────┐
                 │    scale_unit     │   4× (acc·scale_q15 + rnd) >> shift
                 │   ysat[63:0] (4×16)│   saturating to int16
                 └────────┬──────────┘
                          │
                 ┌────────▼──────────┐
                 │  ctrl_fsm drains  │   yw_addr/yw_data/yw_wen
                 └────────┬──────────┘
                          ▼
                        ybuf (port B write; port A read by host)
```

Ports on the top (`t1llm_top`):

| Port | Dir | Width | Meaning |
|------|-----|-------|---------|
| `clk` | in | 1 | system clock |
| `rst_n` | in | 1 | async reset, active low |
| `hif_wr_req` | in | 1 | host write strobe |
| `hif_wr_addr` | in | 16 | word address |
| `hif_wr_data` | in | 32 | write data |
| `hif_rd_req` | in | 1 | host read strobe |
| `hif_rd_addr` | in | 16 | word address |
| `hif_rd_data` | out | 32 | read data (registered) |
| `hif_rd_valid` | out | 1 | read data valid |
| `hif_ready` | out | 1 | accepts transactions |

## 3. Host interface and address map

Synchronous, single-word transactions, always ready. Word-addressed (each address = one 32-bit word).

| Window | Addr[15:12] | Contents |
|--------|-------------|----------|
| Control | `0x0` | registers, see below |
| wmem | `0x1` | ternary weight memory, 4 entries per word (see §4 layout) |
| xbuf | `0x2` | activation buffer, 4 int8 per word |
| ybuf | `0x3` | output buffer, 2 int16 per word (read-only from host) |

Registers (word offsets within the control window):

| Addr | Name | Bits | Meaning |
|------|------|------|---------|
| 0x00 | CTRL | [0] start (self-clearing pulse) · [1] soft_reset · [2] clr_status | |
| 0x01 | STATUS | [0] busy · [1] done · [2] err (read-only) | |
| 0x02 | CFG_K | [15:0] | rows of the weight matrix (= activation length, ≥1) |
| 0x03 | CFG_N | [15:0] | columns (= output length, multiple of 4) |
| 0x04 | SCALE | [15:0] scale_q15 (signed) · [23:16] shift (0..15) | |

## 4. Data formats and memory layout

### 4.1 Ternary weight encoding — 2 bits per weight, 4 per byte

```
bits {sign, nz}:  nz = weight != 0,  sign = weight < 0
  2'b00 →  0     2'b01 → +1     2'b10 → -1     2'b11 → -1
```

Decode in hardware: `value = nz ? (sign ? -1 : +1) : 0`, i.e. a MAC lane either adds, subtracts,
or skips the activation. **No multiplier.**

### 4.2 wmem layout — the GEMV read pattern

`wmem` holds the weight matrix transposed for output-lane streaming. Group output columns into
lanes of 4 (`g = n[15:2]`), with `K` rows each. Entry index `e = g·K + k` is one byte = the four
ternary weights `w[k][4g+0..3]`:

```
byte e = g*K + k:
  bits[1:0]   = w[k][4g+0]     lane 0
  bits[3:2]   = w[k][4g+1]     lane 1
  bits[5:4]   = w[k][4g+2]     lane 2
  bits[7:6]   = w[k][4g+3]     lane 3
```

Total entries = `(N/4)·K`. Host writes 32-bit words = 4 consecutive entries, entry `4w+0` in the
low byte. (First pass uses 8 of every 32 bits at the byte granularity the GEMV stream needs; the
rest of the word is wasted — see §8, packing.)

### 4.3 Activations and outputs

- `xbuf`: `K` signed int8 entries (byte `k`), word = 4 consecutive bytes, byte `4w` in low byte.
- `ybuf`: `N/2` words, each holding two signed int16 outputs, `{y[2g+1], y[2g]}` (hi/lo).

## 5. Datapath and control

### 5.1 GEMV math (exactly what `tools/gen_golden.py` mirrors)

```
acc[n] = Σ_k act[k] · w[k][n]          // ternary w: add/sub/skip, int32 acc
y[n]   = sat16( (acc[n]·scale_q15 + rnd) >>> shift ),  rnd = shift ? (1<<(shift-1)) : 0
```

`>>>` is arithmetic shift; saturation to int16. Integer-only, deterministic, bit-exact.

### 5.2 ctrl_fsm states

```
IDLE → CHECK → ADDR ⇄ MAC (×K) → DRAIN1 → DRAIN2 → (next g | DONE) → IDLE
                └─ invalid config → ERR
```

- **CHECK** — validates `K ≥ 1`, `N % 4 == 0`, `(N/4)·K ≤ WMEM_DEPTH`; clears done/err; asserts busy.
- **ADDR** — drives `wmem.b_addr = g·K + k`, `xbuf.b_addr = k`; asserts `clr_acc` on `k == 0`.
- **MAC** — one cycle later the registered BRAM data is valid; asserts `mac_en` and samples
  `w_in/xb_rdata`. Increments `k`; on `k == K-1` goes to drain.
- **DRAIN1/DRAIN2** — writes `{ysat1,ysat0}` to `ybuf[2g]`, `{ysat3,ysat2}` to `ybuf[2g+1]`.
  Increments `g`; done when `g == N/4`.
- **DONE / ERR** — deasserts busy, latches status. `clr_status` clears it.

Read latency pipeline: registered BRAM reads give `data(addr_t) = data_t+1`; the ADDR/MAC
alternation consumes exactly that.

### 5.3 Throughput (first pass)

`≈ 2·K·N/4` cycles per GEMV (one ADDR + one MAC per (g,k)). Lane-parallelism is 4; the point of
this pass is correctness and a clean wiring skeleton, not throughput. §8 lists the path to a
systolic/tiled engine.

## 6. Verification

- `tools/gen_golden.py <case>` generates golden vectors (hex files) for two randomized cases:
  - case 0: `K=8,  N=16`  (wmem 32 entries, ybuf 8 words; positive scale + rounding path)
  - case 1: `K=16, N=8`   (wmem 32 entries, ybuf 4 words; negative scale + sign path)
- `tb/tb_top.v` drives the design **through the host interface only** (write cfg → write wmem →
  write xbuf → start → poll status → read ybuf), compares against the golden files, and also
  exercises the FSM error path (invalid `N` → `STATUS.err` latched, then cleared via
  `clr_status`). `make sim` must print `=== TB: ALL PASS ===`.
- `make synth` runs yosys `synth_xilinx` as a synthesizability gate (no place-and-route in CI;
  the full nextpnr/prjxray flow lives in `synth/xc7_flow.sh` and runs on the box with the
  toolchain).

## 7. File map

```
hw/1bit-llm/
├── DESIGN.md
├── Makefile
├── rtl/
│   ├── t1llm_top.v     top — wires everything
│   ├── hif_slave.v     host interface, reg file, decode
│   ├── ctrl_fsm.v      GEMV sequencer + status
│   ├── wmem.v          weight BRAM (dual-port, port A host / port B gemv)
│   ├── xbuf.v          activation BRAM (same dual-port shape)
│   ├── ybuf.v          output BRAM (port A host read / port B gemv write)
│   ├── gemv_core.v     4-lane ternary MAC + int32 accumulators
│   └── scale_unit.v    4× saturating q15·acc >> shift
├── tb/tb_top.v
├── tools/gen_golden.py Python bit-exact reference + golden vectors
├── synth/xc7_flow.sh   yosys → nextpnr-xilinx → prjxray → openFPGALoader (stub)
└── sim/                build + generated vectors (gitignored)
```

## 8. Roadmap

1. **Pack wmem to 32-bit words** (4 k-values per word) to remove the 4× waste; add AXI-lite
   host path. *(this pass: plain bus, 8/32 bits used)*
2. **Weight streaming from external DRAM** — the 1B model's weights don't fit on-chip; the
   BRAM cache becomes a double-buffered tile cache.
3. **Systolic/tiled datapath** — N_LANES up to 16-64, K pipelined, multi-bank BRAM; target the
   50-150 tok/s envelope from the journey note.
4. **Full decoder block** — RMSNorm, softmax, KV cache, layer loop; then multi-layer.
5. **Board bring-up** — 7-series (open-source bitstream) first, then the U250 via Vivado using
   this same RTL, plus LUT-LLM-style memory-compute ideas from `research/IDEAS.md` I-07.
