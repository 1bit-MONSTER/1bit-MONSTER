# tq2_aiesim — TQ2 ternary kernel x86sim harness

Single-tile ADF graph running `../mm_ternary_tq2_aie2.cc` on the VEK280
platform in x86sim. `./run.sh` builds, simulates, and checks vs golden.

Status 2026-08-14: harness works at the plumbing level (A=0 → C=0 PASS —
graph wiring, kernel compile, and aiecc link are all sound), but x86sim's
AIE-ML `aie::mmul<4,8,8>` emulation misreads the B operand (single-k probes
return the wrong columns; same class as the STQ sibling's documented
`mac_4x8_8x8` bug that reads only 4 of 64 B bytes). Numerics are verified
by the host mirror test (`../test_tq2_gemv_ref.cc`, max abs err 0.000023
PASS); ground truth for the real kernel binary is AM020 or the physical
board (VE2802 not in aiecompiler's hw device DB in 2026.1 — aiesimulator
unavailable). Keep this harness for the board bring-up flow: swap
`--target=x86sim` for `--target=hw` once the eval kit arrives.

The data generator (`gen_data.cpp`) uses the same seed-42 vectors as the
host mirror test, so golden output and the host reference agree bit-for-bit.
