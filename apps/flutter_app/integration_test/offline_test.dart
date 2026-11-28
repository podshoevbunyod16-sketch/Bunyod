import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('ai.modelforge/offline');

  testWidgets('embedded Android runtime trains and generates without a server', (tester) async {
    final result = await channel.invokeMethod<String>('train', jsonEncode({
      'purpose': 'Тест',
      'text': 'Привет мир. Салом ҷаҳон. Привет мир. Салом ҷаҳон.',
      'target_weight_mb': 0.1,
      'loop_passes': 3,
    }));
    final metadata = jsonDecode(result!) as Map<String, dynamic>;
    expect(metadata['steps'], greaterThan(0));
    expect((metadata['spec'] as Map<String, dynamic>)['loop_passes'], 3);
    final output = await channel.invokeMethod<String>('generate', jsonEncode({
      'prompt': 'Привет', 'max_chars': 4,
    }));
    expect((jsonDecode(output!) as Map<String, dynamic>)['text'], isA<String>());
  }, timeout: const Timeout(Duration(minutes: 4)));
}
