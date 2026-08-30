#!/usr/bin/env bash
# install-docsbot-service.sh — install & start the /docs bot as a systemd user
# service on this host. Idempotent (safe to re-run after updating the bot).
#
# Usage:   ./install-docsbot-service.sh
# Notes:   creates ~/.config/systemd/user/docsbot.service from the template,
#          requires linger so it starts at boot (loginctl enable-linger $USER).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
USER_DIR="$HOME/.config/systemd/user"
SERVICE_NAME="docsbot.service"

echo "==> bot dir: $HERE"
echo "==> ensuring venv + .env exist (secrets live in .env, kept out of git)"
[ -d "$HERE/.venv" ] || python3 -m venv "$HERE/.venv"
"$HERE/.venv/bin/pip" install -q --disable-pip-version-check -r "$HERE/requirements.txt"
[ -f "$HERE/.env" ] || { echo "ERROR: no $HERE/.env — copy .env.example and fill it in"; exit 1; }

echo "==> writing $SERVICE_NAME from template"
mkdir -p "$USER_DIR"
sed -e "s|@BOT_DIR@|$HERE|g" "$HERE/docsbot.service" > "$USER_DIR/$SERVICE_NAME"
chmod 644 "$USER_DIR/$SERVICE_NAME"

echo "==> daemon-reload + enable/start"
systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME" >/dev/null 2>&1 || true

echo "==> enabling linger so the bot starts at boot (no login required)"
loginctl enable-linger "$USER" 2>/dev/null || echo "   (could not enable linger; may need root)"

echo "==> status"
systemctl --user --no-pager status "$SERVICE_NAME" | sed 's/^/   /' | head -12

echo ""
echo "The /docs command is now live in Discord."
echo "   Follow logs:  journalctl --user -u $SERVICE_NAME -f"
echo "   Ask the bot:  /docs <question>"
