#!/usr/bin/env bash
# Build macOS DMG bundling Flutter macOS app + standalone Python Core
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/release_artifacts"
mkdir -p "${OUT_DIR}"

python3 "${REPO_ROOT}/scripts/build_backend_binary.py"
python3 "${REPO_ROOT}/scripts/prepare_flutter_platforms.py"

cd "${REPO_ROOT}/apps/flutter_app"
flutter build macos --release

APP_BUNDLE="${REPO_ROOT}/apps/flutter_app/build/macos/Build/Products/Release/modelforge_app.app"
mkdir -p "${APP_BUNDLE}/Contents/Resources"
rm -rf "${APP_BUNDLE}/Contents/Resources/backend"
cp -r "${OUT_DIR}/backend_bundle/modelforge-backend" "${APP_BUNDLE}/Contents/Resources/backend"

DMG_STAGE="${OUT_DIR}/dmg_stage"
rm -rf "${DMG_STAGE}"
mkdir -p "${DMG_STAGE}"
cp -R "${APP_BUNDLE}" "${DMG_STAGE}/ModelForge AI.app"
ln -s /Applications "${DMG_STAGE}/Applications"

hdiutil create -volname "ModelForge AI" -srcfolder "${DMG_STAGE}" -ov -format UDZO "${OUT_DIR}/ModelForge-macOS.dmg"
rm -rf "${DMG_STAGE}"

cat > "${OUT_DIR}/ModelForge-macOS.SIGNING_STATUS.txt" <<'EOF'
Artifact: ModelForge-macOS.dmg
Signing/Notarization status: Unsigned / Ad-hoc build (Apple Developer ID certificate & notarization credentials are optional and not configured in repository secrets).
EOF

ls -lh "${OUT_DIR}/ModelForge-macOS.dmg"
