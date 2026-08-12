# stq_aiesim — STQ (Sherry 3:4) kernel x86sim harness

Single-tile ADF graph running `../mm_ternary_stq_aie2.cc` on the VEK280
platform in x86sim. `./run.sh` builds, simulates, and checks vs golden.

Status 2026-08-12: harness works (plumbing verified: A=0 → C=0), but x86sim's
AIE-ML `mac_4x8_8x8` int8 emulation reads only 4 of 64 B bytes — emulator bug,
not a real lane geometry. Kernel stays row-major (aie_api convention, iron
aie2p-proven). aiesimulator unavailable: VE2802 not in aiecompiler's hw
device DB in 2026.1. Ground truth: AM020 or the physical board.

Debug modes in gen_data.cpp: `probe <k>` (A-delta at k), `FINGERPRINT=1`
(scales = n+1, outputs self-report their column), `msb` (byte-order flip),
`check` (compare outC.txt vs golden).
