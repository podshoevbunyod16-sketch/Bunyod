#!/usr/bin/env bash
# Build Linux x64 DEB and AppImage bundling Flutter Linux x64 UI + standalone Python Core
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/release_artifacts"
mkdir -p "${OUT_DIR}"

python3 "${REPO_ROOT}/scripts/build_backend_binary.py"
python3 "${REPO_ROOT}/scripts/prepare_flutter_platforms.py"

cd "${REPO_ROOT}/apps/flutter_app"
flutter build linux --release

BUNDLE_DIR="${REPO_ROOT}/apps/flutter_app/build/linux/x64/release/bundle"
rm -rf "${BUNDLE_DIR}/backend"
cp -r "${OUT_DIR}/backend_bundle/modelforge-backend" "${BUNDLE_DIR}/backend"

# 1. Build Debian package (.deb)
DEB_ROOT="${OUT_DIR}/deb_stage"
rm -rf "${DEB_ROOT}"
mkdir -p "${DEB_ROOT}/DEBIAN"
mkdir -p "${DEB_ROOT}/opt/modelforge-ai"
mkdir -p "${DEB_ROOT}/usr/bin"
mkdir -p "${DEB_ROOT}/usr/share/applications"

cp -r "${BUNDLE_DIR}/"* "${DEB_ROOT}/opt/modelforge-ai/"
ln -sf /opt/modelforge-ai/modelforge_app "${DEB_ROOT}/usr/bin/modelforge-ai"
ln -sf /opt/modelforge-ai/backend/modelforge-backend "${DEB_ROOT}/usr/bin/modelforge"

cat > "${DEB_ROOT}/DEBIAN/control" <<'EOF'
Package: modelforge-ai
Version: 1.0.0
Section: science
Priority: optional
Architecture: amd64
Depends: libgtk-3-0, libglib2.0-0, libblkid1, liblzma5
Maintainer: ModelForge AI Team <noreply@modelforge.ai>
Description: ModelForge AI — Cross-platform AI Model, Agent & Research Laboratory
 Unified Material 3 Desktop Studio, Python Core Backend, and CLI.
EOF

cat > "${DEB_ROOT}/usr/share/applications/modelforge-ai.desktop" <<'EOF'
[Desktop Entry]
Name=ModelForge AI
Comment=AI Model, Agent & Research Laboratory
Exec=/opt/modelforge-ai/modelforge_app
Icon=modelforge-ai
Terminal=false
Type=Application
Categories=Development;Science;ArtificialIntelligence;
EOF

dpkg-deb --build --root-owner-group "${DEB_ROOT}" "${OUT_DIR}/ModelForge-Linux-x64.deb"

# 2. Build AppImage
APPDIR="${OUT_DIR}/ModelForge.AppDir"
rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/bin" "${APPDIR}/usr/share/metainfo"
cp -r "${BUNDLE_DIR}/"* "${APPDIR}/usr/bin/"

cat > "${APPDIR}/usr/share/metainfo/modelforge-ai.appdata.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
  <id>modelforge-ai.desktop</id>
  <metadata_license>CC0-1.0</metadata_license>
  <project_license>MIT</project_license>
  <name>ModelForge AI</name>
  <summary>Cross-platform AI Model, Agent and Research Laboratory</summary>
  <developer id="ai.modelforge">
    <name>ModelForge AI</name>
  </developer>
  <description>
    <p>ModelForge AI provides local and hybrid neural model creation, autonomous multi-agent orchestration, scientific experiment tracking, and prompt-driven project editing.</p>
  </description>
  <url type="homepage">https://github.com/podshoevbunyod16-sketch/Bunyod</url>
  <launchable type="desktop-id">modelforge-ai.desktop</launchable>
  <provides>
    <binary>modelforge_app</binary>
  </provides>
  <content_rating type="oars-1.1" />
  <releases>
    <release version="1.0.0" date="2026-11-27" />
  </releases>
</component>
EOF

cat > "${APPDIR}/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "${0}")")"
export LD_LIBRARY_PATH="${HERE}/usr/bin/lib:${LD_LIBRARY_PATH:-}"
exec "${HERE}/usr/bin/modelforge_app" "$@"
EOF
chmod +x "${APPDIR}/AppRun"

cat > "${APPDIR}/modelforge-ai.desktop" <<'EOF'
[Desktop Entry]
Name=ModelForge AI
Exec=modelforge_app
Icon=modelforge-ai
Type=Application
Categories=Development;Science;
EOF

# Minimal valid 1x1 PNG icon for AppDir compliance
python3 - <<PY
import base64, pathlib
png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
pathlib.Path("${APPDIR}/modelforge-ai.png").write_bytes(png)
PY

APPIMAGETOOL="${OUT_DIR}/appimagetool-x86_64.AppImage"
if [ ! -f "${APPIMAGETOOL}" ]; then
  curl -fsSL -o "${APPIMAGETOOL}" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage"
  chmod +x "${APPIMAGETOOL}"
fi

ARCH=x86_64 "${APPIMAGETOOL}" --appimage-extract-and-run --no-appstream "${APPDIR}" "${OUT_DIR}/ModelForge-Linux-x64.AppImage"
chmod +x "${OUT_DIR}/ModelForge-Linux-x64.AppImage"

rm -rf "${DEB_ROOT}" "${APPDIR}" "${APPIMAGETOOL}"
ls -lh "${OUT_DIR}/ModelForge-Linux-x64.deb" "${OUT_DIR}/ModelForge-Linux-x64.AppImage"
