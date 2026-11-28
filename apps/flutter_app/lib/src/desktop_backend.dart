// Desktop & Termux Backend Runtime Manager.
// Never claims Flutter runs Python inside Dart VM; manages bundled native service
// on Windows/Linux/macOS or connects to Termux local daemon / Remote server on Android.

import 'dart:io';

class BackendProcessController {
  Process? _proc;
  bool get isManagedProcessRunning => _proc != null;

  String describePlatformStrategy() {
    if (Platform.isAndroid) {
      return 'Android ARM64: Connect to Termux local daemon (127.0.0.1:8765) or Remote Mode server.';
    }
    return 'Desktop (${Platform.operatingSystem}): Bundled modelforge-backend binary or local Python service.';
  }

  /// Attempts to launch bundled `modelforge-backend` binary adjacent to the desktop executable
  /// if running in Local Mode on Windows/Linux/macOS and backend is not yet listening.
  Future<bool> ensureDesktopLocalBackend({int port = 8765}) async {
    if (Platform.isAndroid || Platform.isIOS) {
      return false;
    }
    final exeDir = File(Platform.resolvedExecutable).parent.path;
    final candidates = Platform.isWindows
        ? <String>[
            '$exeDir\\backend\\modelforge-backend.exe',
            '$exeDir\\modelforge-backend.exe',
          ]
        : <String>[
            '$exeDir/backend/modelforge-backend',
            '$exeDir/modelforge-backend',
            '$exeDir/../Resources/backend/modelforge-backend',
          ];

    for (final bin in candidates) {
      if (await File(bin).exists()) {
        try {
          _proc = await Process.start(
            bin,
            ['server', '--host', '127.0.0.1', '--port', '$port'],
            mode: ProcessStartMode.normal,
          );
          await Future<void>.delayed(const Duration(milliseconds: 900));
          return true;
        } catch (_) {
          return false;
        }
      }
    }
    return false;
  }

  void stop() {
    _proc?.kill();
    _proc = null;
  }
}
