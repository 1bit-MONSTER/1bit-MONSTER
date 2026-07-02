#pragma once
#include <stdint.h>

namespace qwen3 {

// Phase enum (7 phases: Q, K, V, O, UP, GATE, DOWN)
constexpr int32_t kQPhase = 0, kKPhase = 1, kVPhase = 2, kOPhase = 3,
                  kUpPhase = 4, kGatePhase = 5, kDownPhase = 6;
constexpr int32_t kMain16PhaseLimitQkv = kVPhase + 1;      // 3
constexpr int32_t kMain16PhaseLimitQkvo = kOPhase + 1;     // 4
constexpr int32_t kMain16PhaseLimitUpGate = kGatePhase + 1; // 6
constexpr int32_t kMain16PhaseLimitFull = kDownPhase + 1;   // 7

// AIE tile grid
constexpr int32_t kMainRowsPerTile = 32;
constexpr int32_t kQ4GroupSize = 32;
constexpr int32_t kRecordDwords = 17;
constexpr int32_t kRecordPayloadDwords = kRecordDwords - 1;
constexpr int32_t kQ4KChunk = 256;

// Phase body records (PHASE_BLOCKS for 0.6B: Q=4, K=2, V=2, O=2, UP=6, GATE=6, DOWN=2)
constexpr int32_t kQBodyRecords = 4;
constexpr int32_t kKvBodyRecords = 2;
constexpr int32_t kOBodyRecords = 2;
constexpr int32_t kUpGateReplays = 12;    // UP(6) + GATE(6)
constexpr int32_t kDownBodyRecords = 2;

// Chunks per record (PHASE_CHUNKS for 0.6B: Q/K/V/O/UP/GATE=4, DOWN=12)
constexpr int32_t kQChunksPerRecord = 4;
constexpr int32_t kKvChunksPerRecord = 4;
constexpr int32_t kOChunksPerRecord = 4;
constexpr int32_t kUpGateChunksPerReplay = 4;
constexpr int32_t kDownChunksPerRecord = 12;

// Compact record header IDs (same as 8B)
constexpr int32_t kQCompactPacketId = 0x1;
constexpr int32_t kKCompactPacketId = 0x1;
constexpr int32_t kVCompactPacketId = 0x1;
constexpr int32_t kOCompactPacketId = 0x4;
constexpr int32_t kFfnCompactPacketId = 0x8;
constexpr int32_t kDownCompactPacketId = 0x4;

// Lock IDs (same as 8B)
constexpr int32_t kMainActivationEmptyLock = 0, kMainActivationFullLock = 1;
constexpr int32_t kMainWeightEmptyLock = 2, kMainWeightFullLock = 3;
constexpr int32_t kMainRecordEmptyLock = 4, kMainRecordFullLock = 5;
constexpr int32_t kCoreLocalLockBase = 0x30;
constexpr int32_t kMainActivationEmptyCoreLock = kCoreLocalLockBase + kMainActivationEmptyLock;
constexpr int32_t kMainActivationFullCoreLock = kCoreLocalLockBase + kMainActivationFullLock;
constexpr int32_t kMainWeightEmptyCoreLock = kCoreLocalLockBase + kMainWeightEmptyLock;
constexpr int32_t kMainWeightFullCoreLock = kCoreLocalLockBase + kMainWeightFullLock;
constexpr int32_t kMainRecordEmptyCoreLock = kCoreLocalLockBase + kMainRecordEmptyLock;
constexpr int32_t kMainRecordFullCoreLock = kCoreLocalLockBase + kMainRecordFullLock;

// Weight chunk base addresses (per-tile chunks accumulated across phases)
constexpr int32_t kQWeightChunkBase = 0;
constexpr int32_t kKWeightChunkBase = 16;
constexpr int32_t kVWeightChunkBase = 24;
constexpr int32_t kFullLayerOWeightChunkBase = 32;
constexpr int32_t kFullLayerUpGateWeightChunkBase = 40;
constexpr int32_t kFullLayerDownWeightChunkBase = 88;

}  // namespace qwen3
