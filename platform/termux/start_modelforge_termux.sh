#!/data/data/com.termux/files/usr/bin/bash
# Запускает локальный сервер ModelForge Core только на loopback 127.0.0.1:8765
# для безопасного подключения Android Flutter APK на том же устройстве.
set -euo pipefail

export MODELFORGE_LOW_MEMORY="${MODELFORGE_LOW_MEMORY:-1}"
PORT="${MODELFORGE_PORT:-8765}"

echo "Starting ModelForge Core local loopback service on http://127.0.0.1:${PORT} (Low Memory Mode=${MODELFORGE_LOW_MEMORY})..."
if [ -x "$HOME/.modelforge-venv/bin/modelforge" ]; then
  exec "$HOME/.modelforge-venv/bin/modelforge" server --host 127.0.0.1 --port "${PORT}"
else
  exec python3 -m modelforge server --host 127.0.0.1 --port "${PORT}"
fi
