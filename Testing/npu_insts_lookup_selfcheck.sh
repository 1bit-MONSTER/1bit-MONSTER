#!/bin/bash
# npu_insts_lookup_selfcheck.sh — the NPU engine resolves TWO artifacts per
# tensor slot: a compiled xclbin (xp()) and an instruction file (ip()).  If the
# two lookups disagree about which name they accept, a context can pair an
# xclbin found by dimensions with an instruction file found by model tag — or
# find no instruction file at all.
#
# That was a live defect: xp() fell back to a dimension-keyed name
# (final_i8_<t>_K<K>_N<N>.xclbin) but ip() did not (insts_i8_<t>_K<K>_N<N>.txt),
# so a slot whose instruction file is committed under its shape rather than a
# model tag was never found and silently dropped to the runtime generator.  That
# generator emits single-core-row instructions, which per
# init_with_generator's own comment silently computes the WRONG result when
# paired with a multi-row (v27) xclbin.  Qwen3-4B hit it for QKV and O: no
# insts_i8_QKV_qwen3_4b.txt / insts_i8_O_qwen3_4b.txt is committed, while
# insts_i8_QKV_K2560_N6144.txt and insts_i8_O_K4096_N2560.txt both are.
#
# Needs no device, no XRT and no compiler: it reads the real source and the real
# xclbins/ tree, and replays the lookup against the real committed files.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/engine/npu/src/npu_engine_universal.cpp"
DIMS="$ROOT/engine/npu/src/npu_dims.h"
XD="$ROOT/engine/npu/xclbins"
PY="${PYTHON:-python3}"

fails=0
ok()   { echo "  ok   - $*"; }
bad()  { echo "  FAIL - $*"; fails=$((fails+1)); }

for f in "$SRC" "$DIMS" "$XD"; do
    [ -e "$f" ] || { echo "missing input: $f" >&2; exit 2; }
done

# --- Layer 1: the real source must keep the dimension-keyed fallback ----------
# This is what stops a silent revert: the semantic replay below would keep
# passing against a stale assumption, but this reads the shipped .cpp.
echo "source contract (engine/npu/src/npu_engine_universal.cpp)"

# The lambda body is located by NAME.  It used to be pinned to the exact text
# `auto ip=[&](const char*t, int K, int N)`, and #2435 then gave ip() default
# arguments (`int K=-1, int N=-1`) and a trailing return type without touching the
# fallback — so the awk range came back EMPTY and this rule reported "ip() has no
# dimension-keyed fallback (regressed?)" about a file whose fallback was still
# there.  Both halves of Layer 1 were red on main from #2435 until this fix
# (verified: the rule passes at #2435's parent 75b28a8c1 and at the commit that
# introduced it, 02b4e04a3).  A permanently red rule is worse than no rule: the
# real revert it exists to catch is indistinguishable from the false positive.
# Each half now has a control below.
lambda_body() {  # lambda_body <file> <name>
    awk -v name="$2" '
        $0 ~ "^    auto "name"=\\[&\\]" { f=1 }
        f { print }
        f && /^    \};/ { exit }
    ' "$1"
}
have_fallback() {  # have_fallback <file> <name>
    lambda_body "$1" "$2" | grep -vE '^[[:space:]]*//' \
        | grep -qF '"_K"+std::to_string(K)+"_N"'
}

if have_fallback "$SRC" ip; then
    ok "ip() falls back to insts_i8_<t>_K<K>_N<N>.txt"
else
    bad "ip() has no dimension-keyed fallback (regressed?)"
fi

if have_fallback "$SRC" xp; then
    ok "xp() still falls back to final_i8_<t>_K<K>_N<N>.xclbin"
else
    bad "xp() lost its dimension-keyed fallback"
fi

# Every call site must pass the shape, or the fallback is unreachable.  A
# one-argument ip("G") or ip(t) compiles fine and silently restores the old
# behaviour, so match an ip( whose argument list contains no comma.  The lambda
# definition itself is ip=[&](...) and does not match ip\(.  Comments are stripped
# first, respecting string and character literals: the previous grep matched
# `xp()/ip() prefer the` inside a comment at line 2133 and reported prose as a
# call site, which was the other half of the same always-red failure.
onearg_sites() {  # onearg_sites <file>
    "$PY" - "$1" <<'PYEOF'
import re, sys
src = open(sys.argv[1]).read()
out, i, n = [], 0, len(src)
while i < n:                       # strip comments, preserving line numbering
    c = src[i]
    if c in '"\'':
        q = c; i += 1
        while i < n and src[i] != q:
            i += 2 if src[i] == '\\' else 1
        i += 1; out.append(' '); continue
    if src.startswith('//', i):
        while i < n and src[i] != '\n':
            i += 1
        continue
    if src.startswith('/*', i):
        j = src.find('*/', i + 2)
        i = n if j < 0 else j + 2
        out.append(' '); continue
    out.append(c); i += 1
code = ''.join(out)
lines = src.splitlines()
for m in re.finditer(r'(?<![\w])ip\(([^,()]*)\)', code):
    ln = code.count('\n', 0, m.start()) + 1
    print("%d: %s" % (ln, lines[ln - 1].strip()))
PYEOF
}
onearg=$(onearg_sites "$SRC")
if [ -z "$onearg" ]; then
    ok "every ip() call site passes K and N"
else
    bad "an ip() call site passes no shape:"
    printf '        %s\n' "$onearg"
fi

# --- controls: both Layer 1 rules must be able to FAIL ------------------------
# Each rule above went red for a reason unrelated to the code it guards, so each is
# now run against a mutated copy of the same file.  A control that stops holding is
# a finding, not noise: it means the rule has gone blind again.
ctl_d="$(mktemp -d)"
trap 'rm -rf "$ctl_d"' EXIT

# (a) delete ip()'s fallback line — the source rule must notice.
sed 's|return base+"_K"+std::to_string(K)+"_N"+std::to_string(N)+".txt";|return base+"_"+cfg.model_tag+".txt";|' \
    "$SRC" > "$ctl_d/nofallback.cpp"
if [ "$(grep -cF 'return base+"_K"+std::to_string(K)+"_N"+std::to_string(N)+".txt";' "$ctl_d/nofallback.cpp")" -ne 0 ]; then
    bad "control: the mutation did not remove ip()'s fallback line (sed no longer matches the source)"
elif have_fallback "$ctl_d/nofallback.cpp" ip; then
    bad "control: with ip()'s fallback line removed the rule still passes"
else
    ok "control: removing ip()'s fallback line is caught"
fi

# (b) rename the lambda — the name anchor is what makes (a) meaningful.
sed 's|auto ip=\[&\]|auto zz=[\&]|' "$SRC" > "$ctl_d/renamed.cpp"
if have_fallback "$ctl_d/renamed.cpp" ip; then
    bad "control: a renamed ip() lambda still passes the source rule"
else
    ok "control: a renamed ip() lambda is caught"
fi

# (c) a comment naming ip() must not count as a call site (the 2133 false positive).
{ cat "$SRC"; printf '\n// control: xp()/ip() prefer the dimension-keyed name\n'; } \
    > "$ctl_d/withcomment.cpp"
if [ -z "$(onearg_sites "$ctl_d/withcomment.cpp")" ]; then
    ok "control: a comment naming ip() is not read as a call site"
else
    bad "control: a comment naming ip() is still read as a call site:"
    printf '        %s\n' "$(onearg_sites "$ctl_d/withcomment.cpp")"
fi

# --- Layer 2: replay the lookup against the real committed xclbins -----------
# Model dims are parsed out of npu_dims.h, not hardcoded, so this tracks the
# real shapes.  Note what this layer can and cannot catch: it re-implements the
# lookup, so it validates the DATA (which committed names exist for which real
# dims) and proves the bug class was reachable -- but it cannot notice the
# shipped C++ being reverted.  Layer 1 above is what pins the source.
echo "artifact lookup replay (against engine/npu/xclbins)"

replay_out=$("$PY" - "$DIMS" "$XD" <<'PYEOF'
import os, re, sys

dims_path, xd = sys.argv[1], sys.argv[2]

def read_block(path, model):
    """Pull the #define block guarded by #ifdef MODEL_<model>."""
    lines = open(path).read().splitlines()
    start = None
    for i, l in enumerate(lines):
        if re.match(r'\s*#ifdef\s+MODEL_%s\s*$' % re.escape(model), l):
            start = i + 1
            break
    if start is None:
        raise SystemExit("no #ifdef MODEL_%s block in npu_dims.h" % model)
    out = {}
    for l in lines[start:]:
        if re.match(r'\s*#endif', l):
            break
        m = re.match(r'\s*#define\s+(\w+)\s+(.+?)\s*(?://.*)?$', l)
        if m:
            out[m.group(1)] = m.group(2)
    return out

SAFE = re.compile(r'^[0-9+\-*/() ]+$')
def num(block, key):
    v = block[key].strip()
    if SAFE.match(v):
        return int(eval(v, {"__builtins__": {}}, {}))
    raise SystemExit("cannot evaluate %s=%r for the replay" % (key, v))

def resolve(tensor, model_tag, K, N, kn_fallback):
    """Mirror of xp()/ip(): full model tag, then progressively stripped tag,
    then (ip only, once fixed) the dimension-keyed name."""
    base = os.path.join(xd, "insts_i8_" + tensor)
    tag = model_tag
    while True:
        tp = base + "_" + tag + ".txt"
        if os.path.exists(tp):
            return tp
        u = tag.find("_")
        if u == -1 or u == len(tag) - 1:
            break
        tag = tag[u + 1:]
    if kn_fallback:
        kp = base + "_K%d_N%d.txt" % (K, N)
        if os.path.exists(kp):
            return kp
    return base + "_" + model_tag + ".txt"

def slots(block):
    """(tensor, K, N) for the dense i8 slots the engine looks up."""
    H, IM = num(block, "H"), num(block, "IM")
    NH, NKV, HD = num(block, "NH"), num(block, "NKV"), num(block, "HD")
    return [
        ("QKV", H, (NH + 2 * NKV) * HD),
        ("O",   NH * HD, H),
        ("G",   H, IM),
        ("D",   IM, H),
        ("U",   IM, H),
    ]

problems = []

# The premise: the dimension-keyed files this whole check rests on are present.
tag = "qwen3_4b"
block = read_block(dims_path, tag)
mt = block["MODEL_TAG"].strip().strip('"')
must_exist = ["insts_i8_QKV_K2560_N6144.txt", "insts_i8_O_K4096_N2560.txt"]
for name in must_exist:
    if not os.path.exists(os.path.join(xd, name)):
        problems.append("premise gone: %s is not committed" % name)

# The bug: with the fallback missing, qwen3_4b's QKV and O find nothing.
# This is the built-in negative control -- if the replay were trivially
# succeeding it would report no failure here and the check is worthless.
broken_missing = []
fixed_missing = []
changed = []
for tensor, K, N in slots(block):
    old = resolve(tensor, mt, K, N, kn_fallback=False)
    new = resolve(tensor, mt, K, N, kn_fallback=True)
    if not os.path.exists(old):
        broken_missing.append(tensor)
    if not os.path.exists(new):
        fixed_missing.append(tensor)
    if old != new:
        changed.append(tensor)

print("RESULT broken_missing=%s" % ",".join(broken_missing))
print("RESULT fixed_missing=%s" % ",".join(fixed_missing))
print("RESULT changed=%s" % ",".join(changed))

# Negative control: the pre-fix lookup MUST fail on exactly QKV and O. If it
# succeeds everywhere the replay is not exercising the fallback at all.
if sorted(broken_missing) != ["O", "QKV"]:
    problems.append("negative control: expected pre-fix lookup to miss exactly "
                    "QKV,O for %s, got %s" % (tag, broken_missing or "nothing"))
# The fix: both must now resolve to a committed file.
if fixed_missing:
    problems.append("fallback did not resolve %s for %s" % (",".join(fixed_missing), tag))
# Scope: the fix must not change any other slot's resolution.
if sorted(changed) != ["O", "QKV"]:
    problems.append("fallback changed resolution of %s, expected only QKV,O"
                    % (changed or "nothing"))

# No regression: a model WITH tag-named instruction files must keep using them,
# fallback present or not.
tag8 = "qwen3_8b"
b8 = read_block(dims_path, tag8)
mt8 = b8["MODEL_TAG"].strip().strip('"')
if not os.path.exists(os.path.join(xd, "insts_i8_QKV_%s.txt" % mt8)):
    problems.append("premise gone: insts_i8_QKV_%s.txt is not committed" % mt8)
for tensor, K, N in slots(b8):
    old = resolve(tensor, mt8, K, N, kn_fallback=False)
    new = resolve(tensor, mt8, K, N, kn_fallback=True)
    if old != new:
        problems.append("regression: %s %s resolution moved from %s to %s"
                        % (tag8, tensor, os.path.basename(old), os.path.basename(new)))
    if not os.path.exists(new):
        problems.append("%s %s resolves to nothing, fallback or not" % (tag8, tensor))
print("RESULT tag_named_still_preferred=%s" % (tag8,))

if problems:
    for p in problems:
        print("PROBLEM %s" % p)
    raise SystemExit(1)
PYEOF
)
replay_rc=$?

# Echo the replay's findings as individual assertions so a failure names itself.
while IFS= read -r line; do
    case "$line" in
        RESULT\ broken_missing=*) ok "pre-fix lookup misses exactly QKV,O (negative control: ${line#RESULT broken_missing=})" ;;
        RESULT\ fixed_missing=*)  [ -z "${line#RESULT fixed_missing=}" ] && ok "post-fix lookup resolves every slot" ;;
        RESULT\ changed=*)        ok "fallback changes only QKV,O (${line#RESULT changed=})" ;;
        RESULT\ tag_named_still_preferred=*) ok "tag-named files still win for ${line#RESULT tag_named_still_preferred=}" ;;
        PROBLEM\ *) bad "${line#PROBLEM }" ;;
    esac
done <<< "$replay_out"

if [ "$replay_rc" -ne 0 ]; then
    : # problems already reported individually above
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "npu_insts_lookup: all checks passed"
    exit 0
fi
echo "npu_insts_lookup: $fails check(s) failed"
exit 1
