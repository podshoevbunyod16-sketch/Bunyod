#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../apps/flutter_app"
# The training and generation calls must not depend on emulator connectivity.
adb shell svc wifi disable || true
adb shell svc data disable || true
flutter test integration_test/offline_test.dart > "$RUNNER_TEMP/offline-test.log" 2>&1 || {
  cat "$RUNNER_TEMP/offline-test.log"
  details=$(tail -n 60 "$RUNNER_TEMP/offline-test.log" | tr '\n' ' ' | cut -c 1-6000)
  echo "::error::Offline emulator test failed: $details"
  exit 1
}
cat "$RUNNER_TEMP/offline-test.log"
