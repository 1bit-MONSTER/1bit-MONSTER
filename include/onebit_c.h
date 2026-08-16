// onebit_c.h — the one C ABI for the 1bit engine.
//
// This is the ONLY surface Mojo (and anything else) talks to: libonebit.so.
// Thin opaque-handle wrappers over BackendManager; all functions are
// exception-safe (errors surface via onebit_last_error).
//
// Pointer-returning accessors are valid until the next call on the same
// handle. Handles are single-threaded: one handle per inference thread.

#ifndef ONEBIT_C_H
#define ONEBIT_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OneBitHandle OneBitHandle;

/* ── Version ──────────────────────────────────────────────────── */
const char* onebit_version(void);

/* ── Lifecycle ────────────────────────────────────────────────── */
OneBitHandle* onebit_create(void);
void onebit_destroy(OneBitHandle* h);

/* ── Model + hardware init ────────────────────────────────────── */
/* Discover models in weights_dir, pick one (exact model_name match,
 * or the first found if model_name is NULL/empty), scan hardware,
 * and init the best backend. Returns 0 on success, -1 on error. */
int onebit_init(OneBitHandle* h, const char* weights_dir,
                const char* model_name);

/* ── Backends ─────────────────────────────────────────────────── */
int onebit_backend_count(const OneBitHandle* h);
const char* onebit_backend_id(const OneBitHandle* h, int index);
const char* onebit_backend_desc(const OneBitHandle* h, int index);
int onebit_select_backend(OneBitHandle* h, const char* backend_id);

/* ── Inference ────────────────────────────────────────────────── */
/* Run one token; returns the predicted next token id, -1 on error. */
int onebit_generate(OneBitHandle* h, int token_id);
int onebit_reset(OneBitHandle* h);
int onebit_health_check(OneBitHandle* h);

/* ── Errors ───────────────────────────────────────────────────── */
/* Stable until the next call on the same handle. */
const char* onebit_last_error(const OneBitHandle* h);

#ifdef __cplusplus
}
#endif

#endif  // ONEBIT_C_H
