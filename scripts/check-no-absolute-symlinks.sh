#!/usr/bin/env bash
# check-no-absolute-symlinks.sh — Guard against unexpected absolute symlinks
#
# Most absolute symlinks break clones on other machines (see #1043), so they
# are rejected by default. The NPU engine intentionally links model xclbins
# to the local FLM install (/opt/fastflowlm/...), which only exists on the
# deployment box — those targets are allow-listed.
#
# Exit: 0 = clean, 1 = unexpected absolute symlinks found

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

broken=0

while IFS= read -r -d '' f; do
    if [ -L "$f" ]; then
        target=$(readlink "$f")
        case "$target" in
            /*)
                case "$target" in
                    /opt/fastflowlm/*) ;;  # intentional: local FLM NPU install
                    *)
                        echo "ERROR: absolute symlink: $f -> $target"
                        broken=$((broken + 1))
                        ;;
                esac
                ;;
        esac
    fi
done < <(git ls-files -z)

if [ "$broken" -gt 0 ]; then
    echo ""
    echo "Found $broken unexpected git-tracked absolute symlink(s)."
    echo "These break on any clone other than the machine that created them."
    echo "Replace them with real files, relative symlinks, or allow-list the"
    echo "target in this script if it is an intentional machine-local dep."
    exit 1
fi

echo "OK: No unexpected absolute symlinks tracked in git."
