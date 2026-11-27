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

echo "Building Android ARM64 APK (signing_mode=${SIGNING_MODE})..."
flutter build apk --release --target-platform android-arm64

echo "Building Android ARM64 AppBundle (AAB)..."
flutter build appbundle --release --target-platform android-arm64

cp build/app/outputs/flutter-apk/app-release.apk "${OUT_DIR}/ModelForge-Android-ARM64.apk"
cp build/app/outputs/bundle/release/app-release.aab "${OUT_DIR}/ModelForge-Android-ARM64.aab"

cat > "${OUT_DIR}/ModelForge-Android-ARM64.SIGNING_STATUS.txt" <<EOF
Artifact: ModelForge-Android-ARM64.apk / ModelForge-Android-ARM64.aab
Target ABI: arm64-v8a (Android ARM64)
Signing status: ${SIGNING_MODE}
Note: Per ModelForge security specification, production release signing occurs only when ANDROID_KEYSTORE_BASE64, ANDROID_KEYSTORE_PASSWORD, and ANDROID_KEY_ALIAS are configured in GitHub Secrets. No keystore is ever stored in the repository.
EOF

ls -lh "${OUT_DIR}"/ModelForge-Android-ARM64*
