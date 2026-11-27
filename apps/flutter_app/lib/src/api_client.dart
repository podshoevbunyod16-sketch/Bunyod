// REST & WebSocket client for ModelForge Core API (`/api/v1/*` and `/ws/events`)
// with clean Offline Mode enforcement and unreachable-server handling.

import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

class ModelForgeApiException implements Exception {
  final String message;
  final int? statusCode;
  const ModelForgeApiException(this.message, {this.statusCode});

  @override
  String toString() => message;
}

class ModelForgeApiClient {
  String baseUrl;
  String mode; // local | remote | hybrid | offline
  final http.Client _http;
  WebSocketChannel? _ws;

  ModelForgeApiClient({
    this.baseUrl = 'http://127.0.0.1:8765',
    this.mode = 'local',
    http.Client? httpClient,
  }) : _http = httpClient ?? http.Client();

  String get normalizedBase =>
      baseUrl.endsWith('/') ? baseUrl.substring(0, baseUrl.length - 1) : baseUrl;

  void _guardModeForRemote() {
    if (mode == 'offline' &&
        !normalizedBase.contains('127.0.0.1') &&
        !normalizedBase.contains('localhost')) {
      throw const ModelForgeApiException(
        'Offline Mode is active: external network endpoints are blocked.',
      );
    }
  }

  Future<Map<String, dynamic>> getJson(String path) async {
    _guardModeForRemote();
    final uri = Uri.parse('$normalizedBase$path');
    try {
      final resp = await _http.get(uri).timeout(const Duration(seconds: 12));
      final decoded = jsonDecode(utf8.decode(resp.bodyBytes));
      if (resp.statusCode >= 400) {
        final detail = decoded is Map ? (decoded['detail'] ?? decoded['message'] ?? resp.body) : resp.body;
        throw ModelForgeApiException('$detail', statusCode: resp.statusCode);
      }
      return decoded as Map<String, dynamic>;
    } on TimeoutException {
      throw ModelForgeApiException('Backend timeout at $normalizedBase');
    } catch (e) {
      if (e is ModelForgeApiException) rethrow;
      throw ModelForgeApiException('Backend unreachable ($normalizedBase): $e');
    }
  }

  Future<Map<String, dynamic>> postJson(String path, Map<String, dynamic> body) async {
    _guardModeForRemote();
    final uri = Uri.parse('$normalizedBase$path');
    try {
      final resp = await _http
          .post(
            uri,
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 60));
      final decoded = jsonDecode(utf8.decode(resp.bodyBytes));
      if (resp.statusCode >= 400) {
        final detail = decoded is Map ? (decoded['detail'] ?? decoded['message'] ?? resp.body) : resp.body;
        throw ModelForgeApiException('$detail', statusCode: resp.statusCode);
      }
      return decoded as Map<String, dynamic>;
    } on TimeoutException {
      throw ModelForgeApiException('Operation timed out on $normalizedBase');
    } catch (e) {
      if (e is ModelForgeApiException) rethrow;
      throw ModelForgeApiException('Backend request failed ($normalizedBase): $e');
    }
  }

  Stream<Map<String, dynamic>> connectEventStream() {
    _ws?.sink.close();
    final wsBase = normalizedBase
        .replaceFirst('https://', 'wss://')
        .replaceFirst('http://', 'ws://');
    final wsUri = Uri.parse('$wsBase/ws/events');
    _ws = WebSocketChannel.connect(wsUri);
    return _ws!.stream.map((raw) {
      final parsed = jsonDecode(raw.toString());
      return parsed is Map<String, dynamic> ? parsed : <String, dynamic>{'raw': raw.toString()};
    });
  }

  void dispose() {
    _ws?.sink.close();
    _http.close();
  }
}
