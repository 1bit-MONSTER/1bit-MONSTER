// gdn_trace_shim.cpp — LD_PRELOAD live-trace of the GateDeltaNet operator()
// call site in libqwen3_6_moe_npu.so. gdb cannot reach the call (the process
// SIGSEGVs in __memcpy_avx512 during load_weights under gdb's memory layout),
// so we do a runtime patch: the constructor patches the 5-byte `call` at
// offset 0x83437 (the GateDeltaNet npu_app::operator() run) with a `jmp` to a
// naked trampoline that dumps the register-derived buffer pointers, then jumps
// back to 0x8343c so the prefill continues unchanged.
//
// Build: g++ -O2 -fPIC -shared -o libgdn_trace.so gdn_trace_shim.cpp -ldl
// Run:   LD_PRELOAD=./libgdn_trace.so ./moe_prefill_probe
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

static const uintptr_t CALL_OFF = 0x83437;  // call <operator()> (5 bytes)
static const uintptr_t RET_OFF = 0x8343c;   // instruction after the call

extern "C" void gdn_trace_dump(uintptr_t qkv, uintptr_t state, uintptr_t core,
                               uintptr_t rbx, uintptr_t r13, uintptr_t entry_rsp) {
    uintptr_t desc = *(uintptr_t *)(rbx + 0x10);
    uintptr_t wt = *(uintptr_t *)(desc + 0x70);
    int64_t lidx = *(int32_t *)(desc + 0x8);
    uintptr_t wd = *(uintptr_t *)(wt + (size_t)lidx * 8);
    uintptr_t arg5 = *(uintptr_t *)(entry_rsp + 0x8);
    uintptr_t base = *(uintptr_t *)(arg5 + 0x10);
    uintptr_t a = base + (uintptr_t)(int64_t) * (int32_t *)(wd + 0x5d0);
    uintptr_t g = base + (uintptr_t)(int64_t) * (int32_t *)(wd + 0x470);
    uintptr_t b = *(uintptr_t *)(entry_rsp + 0x10);
    uintptr_t ssm_dt = r13;  // r13 = rbx + 0x2d0 (rdi into _activate_a)

    fprintf(stderr, "GDNTRACE qkv=%p state=%p core=%p\n", (void *)qkv, (void *)state, (void *)core);
    fprintf(stderr, "GDNTRACE rbx=%p desc=%p wd=%p base=%p\n", (void *)rbx, (void *)desc, (void *)wd, (void *)base);
    fprintf(stderr, "GDNTRACE a=%p g=%p b=%p ssm_dt=%p\n", (void *)a, (void *)g, (void *)b, (void *)ssm_dt);
    if (a) {
        fprintf(stderr, "GDNTRACE a[0..3]=%g %g %g %g\n", *(float *)a, *(float *)(a + 4), *(float *)(a + 8), *(float *)(a + 12));
    }
    if (b) {
        fprintf(stderr, "GDNTRACE b[0..3]=%g %g %g %g\n", *(float *)b, *(float *)(b + 4), *(float *)(b + 8), *(float *)(b + 12));
    }
    if (ssm_dt) {
        fprintf(stderr, "GDNTRACE ssm_dt[0..3]=%g %g %g %g  [+0x80]=%g %g %g %g\n",
                *(float *)ssm_dt, *(float *)(ssm_dt + 4), *(float *)(ssm_dt + 8), *(float *)(ssm_dt + 12),
                *(float *)(ssm_dt + 0x80), *(float *)(ssm_dt + 0x84), *(float *)(ssm_dt + 0x88), *(float *)(ssm_dt + 0x8c));
    }
    // qkv (conv-qkv bf16) sanity: first 4 bf16
    uint16_t *q = (uint16_t *)qkv;
    fprintf(stderr, "GDNTRACE qkv[0..3]=0x%04x 0x%04x 0x%04x 0x%04x\n", q[0], q[1], q[2], q[3]);
}

// Naked trampoline. Entry via `jmp` (no return address pushed): rsi=conv-qkv,
// rdx=state, rcx=core, rbx=context, r13=rbx+0x2d0. Save caller regs, call dump,
// restore, then tail-jump to RET_OFF (absolute, patched at runtime).
__attribute__((naked, noinline)) void gdn_trampoline() {
    __asm__ volatile(
        "push %rax\n\t"
        "push %rdi\n\t"
        "push %rsi\n\t"
        "push %rdx\n\t"
        "push %rcx\n\t"
        "push %r8\n\t"
        "push %r9\n\t"
        "push %r10\n\t"
        "push %r11\n\t"
        "push %rbx\n\t"
        "push %r13\n\t"
        "sub $0x28, %rsp\n\t"        // 11*8 + 0x28 = 0x80 total; keep 16-byte align
        "mov %rsi, %rdi\n\t"         // arg1 = qkv
        "mov %rdx, %rsi\n\t"         // arg2 = state
        "mov %rcx, %rdx\n\t"         // arg3 = core
        "mov %rbx, %rcx\n\t"         // arg4 = rbx (context)
        "mov %r13, %r8\n\t"          // arg5 = r13 (ssm_dt)
        "lea 0x80(%rsp), %r9\n\t"    // arg6 = entry rsp (11 pushes + 0x28)
        "call gdn_trace_dump\n\t"
        "add $0x28, %rsp\n\t"
        "pop %r13\n\t"
        "pop %rbx\n\t"
        "pop %r11\n\t"
        "pop %r10\n\t"
        "pop %r9\n\t"
        "pop %r8\n\t"
        "pop %rcx\n\t"
        "pop %rdx\n\t"
        "pop %rsi\n\t"
        "pop %rdi\n\t"
        "pop %rax\n\t"
        "movabs $0x123456789abcdef0, %rax\n\t"  // patched to RET_OFF abs addr
        "push %rax\n\t"
        "ret\n\t");
}

__attribute__((constructor)) static void gdn_trace_init() {
    uintptr_t lib_base = 0;
    FILE *f = fopen("/proc/self/maps", "r");
    char line[512];
    while (f && fgets(line, sizeof line, f)) {
        if (strstr(line, "libqwen3_6_moe_npu.so") && strstr(line, "r-xp")) {
            sscanf(line, "%lx-", &lib_base);
            break;
        }
    }
    if (f) fclose(f);
    if (!lib_base) {
        fprintf(stderr, "GDNTRACE: lib not found in maps\n");
        return;
    }
    fprintf(stderr, "GDNTRACE: lib base = %p\n", (void *)lib_base);

    uintptr_t call_addr = lib_base + CALL_OFF;
    uintptr_t ret_addr = lib_base + RET_OFF;

    // make the shim's own .text page writable before patching the movabs
    // immediate (RELRO makes .text read-only -> SIGSEGV otherwise)
    uintptr_t tpage = (uintptr_t)&gdn_trampoline & ~(uintptr_t)0xfff;
    mprotect((void *)tpage, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC);

    // patch the trampoline's movabs immediate (48 b8 imm64) with ret_addr
    uint8_t *t = (uint8_t *)&gdn_trampoline;
    for (size_t i = 0; i + 10 < 160; i++) {
        if (t[i] == 0x48 && t[i + 1] == 0xb8) {
            uintptr_t imm = *(uintptr_t *)(t + i + 2);
            if (imm == 0x123456789abcdef0ULL) {
                *(uintptr_t *)(t + i + 2) = ret_addr;
                break;
            }
        }
    }

    // patch the call site: 5-byte call -> jmp rel32 to the trampoline
    uintptr_t page = call_addr & ~(uintptr_t)0xfff;
    mprotect((void *)page, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC);
    int32_t rel = (int32_t)((uintptr_t)&gdn_trampoline - (call_addr + 5));
    uint8_t patch[5];
    patch[0] = 0xe9;  // jmp rel32
    memcpy(patch + 1, &rel, 4);
    memcpy((void *)call_addr, patch, 5);
    fprintf(stderr, "GDNTRACE: patched 0x%lx -> trampoline (ret 0x%lx)\n", call_addr, ret_addr);
}
