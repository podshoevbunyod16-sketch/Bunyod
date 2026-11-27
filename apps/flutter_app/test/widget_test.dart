import 'dart:convert';
import 'dart:ui' show Size;
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:modelforge_app/main.dart';
import 'package:modelforge_app/src/api_client.dart';

void main() {
  testWidgets('ModelForge Flutter GUI renders Dashboard, navigates workspaces & switches Tajik/Offline modes',
      (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({});

    final mockHttp = MockClient((http.Request req) async {
      if (req.url.path == '/api/v1/doctor') {
        return http.Response(
          jsonEncode({
            'healthy': true,
            'mode': 'local',
            'platform': {'low_memory_mode': true},
            'counts': {'projects': 2, 'models': 3, 'agents': 1, 'experiments': 4},
          }),
          200,
          headers: {'content-type': 'application/json'},
        );
      }
      return http.Response(
        jsonEncode({
          'projects': [],
          'models': [],
          'agents': [],
          'tasks': [],
          'experiments': [],
          'servers': [],
          'plugins': [],
          'providers': [],
        }),
        200,
        headers: {'content-type': 'application/json'},
      );
    });

    final client = ModelForgeApiClient(
      baseUrl: 'http://127.0.0.1:8765',
      httpClient: mockHttp,
    );

    tester.view.physicalSize = const Size(1280, 1080);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(
      ModelForgeApp(
        injectedClient: client,
        autoConnectOnStart: false,
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('ModelForge AI'), findsOneWidget);
    expect(find.text('Панель'), findsOneWidget);

    // Navigate to Settings and switch UI language to Tajik ('Тоҷикӣ')
    await tester.tap(find.text('Настройки'));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Тоҷикӣ'));
    await tester.pumpAndSettle();

    expect(find.text('Лавҳаи асосӣ'), findsOneWidget);
    expect(find.text('Омӯзиш'), findsOneWidget);

    // Switch to Offline Mode and verify explicit Offline banner appears
    await tester.tap(find.text('Offline'));
    await tester.pumpAndSettle();
    expect(find.textContaining('офлайн'), findsOneWidget);
  });
}
