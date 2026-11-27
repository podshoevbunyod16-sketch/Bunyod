#!/data/data/com.termux/files/usr/bin/bash
# ModelForge AI — Non-root Termux ARM64 installer (Android 10+ / 3 GB+ RAM)
set -euo pipefail

echo "=== ModelForge AI: Установка в Termux ARM64 (без root) ==="
pkg update -y
pkg install -y python python-numpy git clang make libffi openssl

python -m venv "$HOME/.modelforge-venv"
source "$HOME/.modelforge-venv/bin/activate"
pip install --upgrade pip setuptools wheel
pip install pydantic pyyaml fastapi uvicorn httpx websockets safetensors
pip install --no-deps -e .

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/modelforge" <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
export MODELFORGE_LOW_MEMORY=1
exec "$HOME/.modelforge-venv/bin/modelforge" "$@"
EOF
chmod +x "$HOME/.local/bin/modelforge"

echo "=== Проверка установки (modelforge doctor) ==="
"$HOME/.local/bin/modelforge" doctor
echo "Готово! Добавьте ~/.local/bin в PATH и запустите: bash platform/termux/start_modelforge_termux.sh"
