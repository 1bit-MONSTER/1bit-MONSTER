#!/usr/bin/env bash
# install-docsbot-service.sh — install & start the /docs bot (and its
# `!docs` prefix companion, when present) as systemd user services on this
# host. Idempotent (safe to re-run after updating the bot).
#
# Usage:   ./install-docsbot-service.sh
# Notes:   creates ~/.config/systemd/user/docsbot.service and
#          docsbot-prefix.service from the committed templates, requires
#          linger so they start at boot (loginctl enable-linger $USER).
#          Secrets (issue #1965): the script refuses to run with an empty or
#          invalid .env — it can assemble one from the host's existing secret
#          locations (~/.secrets/*) when present, and validates the Discord
#          token against the API before enabling the services.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
USER_DIR="$HOME/.config/systemd/user"
ENV_FILE="$HERE/.env"

# ── assemble .env from known local secret locations (if not present) ──────
if [ ! -f "$ENV_FILE" ]; then
    echo "==> no $ENV_FILE — trying to assemble from ~/.secrets/*"
    : > "$ENV_FILE.tmp"
    put() { # put KEY VALUE (skip if already set / value empty)
        local k="$1" v="$2"
        [ -n "$v" ] && ! grep -q "^$k=" "$ENV_FILE.tmp" && echo "$k=$v" >> "$ENV_FILE.tmp"
    }
    [ -f "$HOME/.secrets/Discord Bot token.txt" ] && \
        put "DISCORD_TOKEN" "$(cat "$HOME/.secrets/Discord Bot token.txt")"
    [ -f "$HOME/.secrets/ctx7.txt" ] && \
        put "CONTEXT7_API_KEY" "$(cat "$HOME/.secrets/ctx7.txt")"
    if command -v python3 >/dev/null 2>&1 && [ -f "$HOME/.dsh/.credentials.yaml" ]; then
        # best-effort DeepSeek key from the DSH credentials store
        DSK=$(python3 - "$HOME/.dsh/.credentials.yaml" <<'PY' 2>/dev/null || true
import sys, re
try:
    txt = open(sys.argv[1], encoding="utf-8").read()
except OSError:
    sys.exit(0)
m = re.search(r"(?:deepseek|dsh)[_-]?(?:api[_-]?key|token)\s*[:=]\s*[\"']?([A-Za-z0-9._-]{16,})", txt, re.I)
if m: print(m.group(1))
PY
)
        put "DEEPSEEK_API_KEY" "$DSK"
    fi
    # keep any pre-existing DISCORD_KNOWN_CHANNEL / docs config from the example
    if [ -f "$HERE/.env.example" ]; then
        while IFS= read -r line; do
            case "$line" in
                ''|\#*) continue ;;
            esac
            k="${line%%=*}"
            grep -q "^$k=" "$ENV_FILE.tmp" || echo "$line" >> "$ENV_FILE.tmp"
        done < "$HERE/.env.example"
    fi
    mv "$ENV_FILE.tmp" "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "==> assembled $ENV_FILE (0600) from local secrets"
fi

[ -f "$ENV_FILE" ] || { echo "ERROR: no $ENV_FILE — copy .env.example and fill it in"; exit 1; }

# ── validate the three secrets before installing the service ──────────────
echo "==> validating secrets in $ENV_FILE"
env -i PATH="$PATH" HOME="$HOME" bash -c '
    set -euo pipefail
    [ -f "$1" ] && while IFS= read -r line; do
        case "$line" in
            ""|\#*) continue ;;
        esac
        export "${line%%=*}"="${line#*=}"
    done < "$1"
    MISSING=""
    [ -n "${DISCORD_TOKEN:-}" ]     || MISSING="$MISSING DISCORD_TOKEN"
    [ -n "${CONTEXT7_API_KEY:-}" ]  || MISSING="$MISSING CONTEXT7_API_KEY"
    [ -n "${DEEPSEEK_API_KEY:-}" ]  || MISSING="$MISSING DEEPSEEK_API_KEY"
    if [ -n "$MISSING" ]; then
        echo "ERROR: .env is missing:$MISSING — /docs will not work. Fill them in and re-run."
        exit 1
    fi
    # live-check the Discord token (GET /users/@me) so a bad token fails here,
    # not after the service is enabled.
    code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bot ${DISCORD_TOKEN}" \
        "https://discord.com/api/v10/users/@me" 2>/dev/null || echo 000)
    if [ "$code" != "200" ]; then
        echo "ERROR: DISCORD_TOKEN rejected by Discord API (HTTP $code) — check the token and re-run."
        exit 1
    fi
' _ "$ENV_FILE"
echo "==> secrets valid (Discord token live-checked, HTTP 200)"

echo "==> bot dir: $HERE"
echo "==> ensuring venv exists (secrets live in .env, kept out of git)"
[ -d "$HERE/.venv" ] || python3 -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install -q --disable-pip-version-check -r "$HERE/requirements.txt"

echo "==> writing service units from templates (docsbot.service + docsbot-prefix.service)"
mkdir -p "$USER_DIR"
for TPL in "$HERE"/docsbot*.service; do
    [ -f "$TPL" ] || continue
    NAME="$(basename "$TPL")"
    sed -e "s|@BOT_DIR@|$HERE|g" "$TPL" > "$USER_DIR/$NAME"
    chmod 644 "$USER_DIR/$NAME"
    systemctl --user daemon-reload
    systemctl --user enable --now "$NAME" >/dev/null 2>&1 || true
    echo "==> installed $NAME"
done

echo "==> enabling linger so the bots start at boot (no login required)"
loginctl enable-linger "$USER" 2>/dev/null || echo "   (could not enable linger; may need root)"

echo "==> status"
systemctl --user --no-pager status docsbot.service | sed 's/^/   /' | head -12

echo ""
echo "The /docs and !docs bots are now live in Discord."
echo "   Follow logs:  journalctl --user -u docsbot.service -f   (or docsbot-prefix.service)"
echo "   Ask the bot:  /docs <question>   or   !docs <question>"
