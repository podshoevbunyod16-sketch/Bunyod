#!/usr/bin/env bash
# Build Android ARM64 APK and AAB artifacts via Flutter & Gradle
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/release_artifacts"
mkdir -p "${OUT_DIR}"

python3 "${REPO_ROOT}/scripts/prepare_flutter_platforms.py"

cd "${REPO_ROOT}/apps/flutter_app"

SIGNING_MODE="debug_test_key_not_play_store_signed"
if [ -n "${ANDROID_KEYSTORE_BASE64:-}" ] && [ -n "${ANDROID_KEY_ALIAS:-}" ] && [ -n "${ANDROID_KEYSTORE_PASSWORD:-}" ]; then
  echo "${ANDROID_KEYSTORE_BASE64}" | base64 --decode > android/app/upload-keystore.jks
  cat > android/key.properties <<EOF
storePassword=${ANDROID_KEYSTORE_PASSWORD}
keyPassword=${ANDROID_KEY_PASSWORD:-$ANDROID_KEYSTORE_PASSWORD}
keyAlias=${ANDROID_KEY_ALIAS}
storeFile=upload-keystore.jks
EOF
  SIGNING_MODE="release_keystore_from_github_secrets"
fi

echo "Building Android ARM64 + ARMv7 universal-ARM APK (signing_mode=${SIGNING_MODE})..."
flutter build apk --release --target-platform android-arm,android-arm64

echo "Building Android ARM64 AppBundle (AAB)..."
flutter build appbundle --release --target-platform android-arm,android-arm64

cp build/app/outputs/flutter-apk/app-release.apk "${OUT_DIR}/ModelForge-Android-ARM64.apk"
cp build/app/outputs/bundle/release/app-release.aab "${OUT_DIR}/ModelForge-Android-ARM64.aab"

# Verify APK signature (v1 + v2/v3) so Android Package Installer installs cleanly
APKSIGNER="$(find "${ANDROID_HOME:-/usr/local/lib/android/sdk}/build-tools" -name apksigner 2>/dev/null | sort -V | tail -n 1 || true)"
if [ -n "${APKSIGNER}" ] && [ -x "${APKSIGNER}" ]; then
  "${APKSIGNER}" verify --verbose "${OUT_DIR}/ModelForge-Android-ARM64.apk" | head -n 12
fi

cat > "${OUT_DIR}/ModelForge-Android-ARM64.SIGNING_STATUS.txt" <<EOF
Artifact: ModelForge-Android-ARM64.apk / ModelForge-Android-ARM64.aab
Target ABIs: arm64-v8a + armeabi-v7a (compatible with both 64-bit ARM64 and 32-bit Zygote mode Android devices)
Signing status: ${SIGNING_MODE}
Note: Per ModelForge security specification, production release signing occurs only when ANDROID_KEYSTORE_BASE64, ANDROID_KEYSTORE_PASSWORD, and ANDROID_KEY_ALIAS are configured in GitHub Secrets. Without secrets, APK is signed with standard Android debug key for direct sideloading ("Install unknown apps"). No keystore is ever stored in the repository.
EOF

ls -lh "${OUT_DIR}"/ModelForge-Android-ARM64*
