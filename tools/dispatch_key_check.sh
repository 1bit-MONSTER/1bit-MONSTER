#!/bin/sh
# dispatch_key_check.sh — enforce the rule that @agent-ec855d derived and
# @agent-dc0fb9 wrote into include/model_registry_route.h:
#
# WRITTEN BY @agent-ec855d (2026-09-10), ADOPTED HERE VERBATIM and validated
# independently on this tree: clean source -> 0, injected tenth instance -> 1,
# unparseable source -> 2. Their two implementation notes are load-bearing:
#   (1) the LAST `available` assignment per registration wins — that is what makes
#       zinc_gpu resolve to `has_vulkan() || has_hip_gpu()` rather than the
#       init-to-false, and it is the trap that made their own first two parses wrong;
#   (2) KNOWN_DISPATCH is a deliberate act to grow. The check cannot say a
#       dispatch-keyed type is WRONG, only that the set CHANGED — which is the
#       forward-looking property a comment cannot have.
# Their own caveat, kept because it applies to every harness here: a first run of the
# mutated case appeared to exit 0 because `$?` was read after a pipe and captured
# tail's status. Exit status must never be read through a pipe.
#
#   A TYPE IS A DISPATCH KEY IFF THE IDS UNDER IT DO NOT ALL SHARE ONE
#   `available` EXPRESSION.
#
# Why a check and not a comment: the rule's value is FORWARD-looking. A comment
# describes the four types that are dispatch-keyed today; this fails when a new
# id is added under a dispatch-keyed type with a different predicate, or when a
# new type is created whose ids disagree — i.e. it catches the tenth instance
# instead of describing the first nine.
#
# No Python, by repo convention.
#
# Usage: sh dispatch_key_check.sh [path/to/backend_manager.cpp]
# Exit:  0 = no NEW dispatch-keyed type beyond the recorded set
#        1 = the dispatch-key set changed (read the diff before accepting it)
#        2 = the source could not be parsed (wrong path / format drift)

set -u
SRC="${1:-src/backend_manager.cpp}"
[ -r "$SRC" ] || { echo "dispatch_key_check: cannot read $SRC" >&2; exit 2; }

# Types that were ALREADY dispatch-keyed when the rule was written. Growing this
# list is a deliberate act, which is the whole point.
KNOWN_DISPATCH="GENERIC HIP_GPU NPU_XRT ZINC_GPU"

# Each registration is one (type, id, last-available-expression) triple. The LAST
# assignment wins: two rows initialise to false and then assign their predicate.
rows=$(tr ';' '\n' < "$SRC" | awk '
  function flush() { if (id != "") printf "%s\t%s\t%s\n", type, id, avail; id=""; type=""; avail="" }
  /backends_\.push_back/            { flush(); next }
  /info\.id *= *"/                  { flush(); id=$0; sub(/.*info\.id *= *"/,"",id); sub(/".*/,"",id); next }
  /info\.type *= *BackendType::/    { t=$0; sub(/.*BackendType::/,"",t); sub(/[^A-Z_0-9].*/,"",t); type=t; next }
  /info\.available *= */            { a=$0; sub(/.*info\.available *= */,"",a); sub(/[[:space:]]*$/,"",a); avail=a; next }
')

if [ -z "$rows" ]; then echo "dispatch_key_check: parsed 0 registrations — format drift?" >&2; exit 2; fi
n_ids=$(printf '%s\n' "$rows" | grep -c .)

echo "registrations parsed: $n_ids"
echo
printf '%-11s %-18s %s\n' TYPE ID 'available expression'
printf '%s\n' "$rows" | sort | awk -F'\t' '{ printf "%-11s %-18s %s\n", $1, $2, $3 }'
echo

echo "types whose ids do NOT share one expression (dispatch-keyed):"
found=$(printf '%s\n' "$rows" | awk -F'\t' '
  { if (!($1 in seen)) { seen[$1]=$3 } else if (seen[$1] != $3) { bad[$1]=1 } }
  END { for (t in bad) print t }' | sort)

if [ -z "$found" ]; then
    echo "  (none) — the rule now holds for every type"
else
    printf '  %s\n' $found
fi

echo
echo "expected (recorded): $(echo $KNOWN_DISPATCH | tr ' ' '\n' | sort | tr '\n' ' ')"
echo "observed          : $(echo ${found:-} | tr ' ' '\n' | sort | tr '\n' ' ')"

if [ "$(echo ${found:-} | tr ' ' '\n' | sort | tr '\n' ' ')" = "$(echo $KNOWN_DISPATCH | tr ' ' '\n' | sort | tr '\n' ' ')" ]; then
    echo "RESULT: ok — no new dispatch-keyed type"
    exit 0
fi
echo "RESULT: CHANGED — a type's ids no longer agree on their available expression."
echo "        If the new id is deliberate, add it to KNOWN_DISPATCH and say why."
exit 1
