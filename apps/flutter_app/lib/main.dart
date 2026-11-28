// ModelForge AI — Adaptive Material 3 GUI for Android ARM64, Windows x64, Linux x64 & macOS.
// Implements all 12 workspaces specified in Section 12 of the Master Specification.

import 'dart:async';
import 'dart:convert';
import 'dart:io' show Platform;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'src/api_client.dart';
import 'src/desktop_backend.dart';
import 'src/localization.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ModelForgeApp());
}

class ModelForgeApp extends StatefulWidget {
  final ModelForgeApiClient? injectedClient;
  final bool autoConnectOnStart;

  const ModelForgeApp({
    super.key,
    this.injectedClient,
    this.autoConnectOnStart = true,
  });

  @override
  State<ModelForgeApp> createState() => _ModelForgeAppState();
}

class _ModelForgeAppState extends State<ModelForgeApp> {
  ThemeMode _themeMode = ThemeMode.dark;
  String _localeCode = 'ru';
  double _textScale = 1.0;

  @override
  void initState() {
    super.initState();
    _loadPrefs();
  }

  Future<void> _loadPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      _localeCode = prefs.getString('mf_locale') ?? 'ru';
      _textScale = prefs.getDouble('mf_text_scale') ?? 1.0;
      final themeStr = prefs.getString('mf_theme') ?? 'dark';
      _themeMode = themeStr == 'light'
          ? ThemeMode.light
          : (themeStr == 'system' ? ThemeMode.system : ThemeMode.dark);
    });
  }

  Future<void> _updateSettings({
    ThemeMode? themeMode,
    String? localeCode,
    double? textScale,
  }) async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      if (themeMode != null) {
        _themeMode = themeMode;
        prefs.setString('mf_theme', themeMode.name);
      }
      if (localeCode != null) {
        _localeCode = localeCode;
        prefs.setString('mf_locale', localeCode);
      }
      if (textScale != null) {
        _textScale = textScale;
        prefs.setDouble('mf_text_scale', textScale);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final strings = AppStrings(_localeCode);
    final flutterLocale = _localeCode == 'en' ? const Locale('en') : const Locale('ru');

    return MaterialApp(
      title: strings.tr('app_title'),
      debugShowCheckedModeBanner: false,
      themeMode: _themeMode,
      locale: flutterLocale,
      supportedLocales: const [Locale('ru'), Locale('en')],
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0284C7),
          brightness: Brightness.light,
        ),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF38BDF8),
          brightness: Brightness.dark,
        ),
      ),
      builder: (context, child) {
        return MediaQuery(
          data: MediaQuery.of(context).copyWith(
            textScaler: TextScaler.linear(_textScale),
          ),
          child: child ?? const SizedBox.shrink(),
        );
      },
      home: ModelForgeShell(
        strings: strings,
        themeMode: _themeMode,
        localeCode: _localeCode,
        textScale: _textScale,
        onSettingsChanged: _updateSettings,
        injectedClient: widget.injectedClient,
        autoConnectOnStart: widget.autoConnectOnStart,
      ),
    );
  }
}

class NavSection {
  final String keyName;
  final IconData icon;
  const NavSection(this.keyName, this.icon);
}

const List<NavSection> kSections = [
  NavSection('dashboard', Icons.space_dashboard_outlined),
  NavSection('chat', Icons.chat_bubble_outline),
  NavSection('model_studio', Icons.hub_outlined),
  NavSection('agent_studio', Icons.smart_toy_outlined),
  NavSection('training', Icons.model_training_outlined),
  NavSection('research', Icons.science_outlined),
  NavSection('explorer', Icons.difference_outlined),
  NavSection('registry', Icons.inventory_2_outlined),
  NavSection('mcp_plugins', Icons.extension_outlined),
  NavSection('providers', Icons.vpn_key_outlined),
  NavSection('monitor', Icons.monitor_heart_outlined),
  NavSection('settings', Icons.settings_outlined),
];

class ModelForgeShell extends StatefulWidget {
  final AppStrings strings;
  final ThemeMode themeMode;
  final String localeCode;
  final double textScale;
  final void Function({ThemeMode? themeMode, String? localeCode, double? textScale})
      onSettingsChanged;
  final ModelForgeApiClient? injectedClient;
  final bool autoConnectOnStart;

  const ModelForgeShell({
    super.key,
    required this.strings,
    required this.themeMode,
    required this.localeCode,
    required this.textScale,
    required this.onSettingsChanged,
    this.injectedClient,
    this.autoConnectOnStart = true,
  });

  @override
  State<ModelForgeShell> createState() => _ModelForgeShellState();
}

class _ModelForgeShellState extends State<ModelForgeShell> {
  int _selectedIndex = 0;
  late final ModelForgeApiClient _api;
  final BackendProcessController _backendProc = BackendProcessController();
  StreamSubscription<Map<String, dynamic>>? _wsSub;

  bool _backendConnected = false;
  bool _lowMemoryMode = true;
  bool _busy = false;
  String _statusBanner = '';
  String _activeProjectId = '';
  String _lastEditId = '';

  Map<String, dynamic> _doctor = const {};
  List<dynamic> _projects = const [];
  List<dynamic> _models = const [];
  List<dynamic> _agents = const [];
  List<dynamic> _tasks = const [];
  List<dynamic> _experiments = const [];
  List<dynamic> _mcpServers = const [];
  List<dynamic> _plugins = const [];
  List<dynamic> _providers = const [];
  final List<String> _liveEvents = [];

  // Offline on-device project cache so Offline Mode works even when no Python daemon is running
  final List<Map<String, dynamic>> _localDeviceProjects = [];

  // Form controllers
  final TextEditingController _serverUrlCtrl =
      TextEditingController(text: 'http://127.0.0.1:8765');
  final TextEditingController _projectNameCtrl =
      TextEditingController(text: 'RU-TJ Compact Lab');
  final TextEditingController _chatCtrl = TextEditingController(
      text: 'Салом! Привет! Оцени ресурсы для компактной модели.');
  final TextEditingController _modelPromptCtrl = TextEditingController(
      text:
          'Создай компактную языковую модель для русского и таджикского текста');
  final TextEditingController _inferModelIdCtrl = TextEditingController();
  final TextEditingController _inferPromptCtrl =
      TextEditingController(text: 'Салом ҷаҳон! Привет');
  final TextEditingController _agentPromptCtrl = TextEditingController(
      text:
          'Исследовательский агент, который изучает проект и запускает безопасные тесты маленьких моделей');
  final TextEditingController _agentTaskCtrl =
      TextEditingController(text: 'Проверь архитектуру и файлы рабочей директории');
  final TextEditingController _trainEpochsCtrl = TextEditingController(text: '4');
  final TextEditingController _trainLrCtrl = TextEditingController(text: '0.004');
  String _trainMode = 'scratch';
  int _loopPasses = 1;
  double _targetWeightMb = 1.0;
  final _purposeCtrl = TextEditingController();
  final _dataCtrl = TextEditingController();
  String _quickStatus = '';
  final _localPromptCtrl = TextEditingController();
  String _localAnswer = '';
  static const _offlineChannel = MethodChannel('ai.modelforge/offline');
  final TextEditingController _researchQuestionCtrl = TextEditingController(
      text: 'Сравнение decoder_transformer и hybrid_attention_ssm на RU-TJ корпусе');
  final TextEditingController _promptEditCtrl =
      TextEditingController(text: 'Добавь таджикский язык и снизь RAM');
  final TextEditingController _registryQueryCtrl = TextEditingController();

  String _outputConsole = 'ModelForge AI готов к работе.';

  @override
  void initState() {
    super.initState();
    _api = widget.injectedClient ?? ModelForgeApiClient();
    if (widget.autoConnectOnStart && !Platform.isAndroid) {
      _bootstrapConnection();
    }
  }

  @override
  void dispose() {
    _wsSub?.cancel();
    if (widget.injectedClient == null) {
      _api.dispose();
    }
    _backendProc.stop();
    _serverUrlCtrl.dispose();
    _projectNameCtrl.dispose();
    _chatCtrl.dispose();
    _modelPromptCtrl.dispose();
    _inferModelIdCtrl.dispose();
    _inferPromptCtrl.dispose();
    _agentPromptCtrl.dispose();
    _agentTaskCtrl.dispose();
    _trainEpochsCtrl.dispose();
    _trainLrCtrl.dispose();
    _purposeCtrl.dispose();
    _dataCtrl.dispose();
    _localPromptCtrl.dispose();
    _researchQuestionCtrl.dispose();
    _promptEditCtrl.dispose();
    _registryQueryCtrl.dispose();
    super.dispose();
  }

  Future<void> _bootstrapConnection() async {
    await _refreshAll();
    if (!mounted || _backendConnected) return;
    final started = await _backendProc.ensureDesktopLocalBackend();
    if (started && mounted) await _refreshAll();
  }

  void _connectWs() {
    _wsSub?.cancel();
    try {
      _wsSub = _api.connectEventStream().listen(
        (evt) {
          if (!mounted) return;
          setState(() {
            final line = '[${evt['topic'] ?? 'event'}] ${jsonEncode(evt['payload'] ?? evt)}';
            _liveEvents.insert(0, line);
            if (_liveEvents.length > 40) {
              _liveEvents.removeLast();
            }
          });
        },
        onError: (_) {},
      );
    } catch (_) {}
  }

  Future<void> _runGuarded(String label, Future<void> Function() action) async {
    setState(() {
      _busy = true;
      _statusBanner = '$label...';
    });
    try {
      await action();
      if (mounted) {
        setState(() => _statusBanner = '$label: OK');
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          if (e is ModelForgeApiException && e.statusCode == null) {
            _backendConnected = false;
            _statusBanner = Platform.isAndroid
                ? 'Не удалось подключиться к вычислительному ядру. '
                    'В этой версии Android обучение внутри приложения ещё не поддерживается.'
                : 'Не удалось подключиться к вычислительному ядру. Проверьте, запущено ли оно.';
          } else {
            _statusBanner = 'Ошибка: $e';
          }
          _outputConsole = '[$label] Ошибка:\n$e';
        });
      }
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  Future<void> _refreshAll() async {
    await _runGuarded('Синхронизация с ModelForge Core', () async {
      final doc = await _api.getJson('/api/v1/doctor');
      final projs = await _api.getJson('/api/v1/projects');
      final mdls = await _api.getJson('/api/v1/models');
      final agts = await _api.getJson('/api/v1/agents');
      final tsks = await _api.getJson('/api/v1/tasks');
      final exps = await _api.getJson('/api/v1/experiments');
      final mcps = await _api.getJson('/api/v1/mcp');
      final plugs = await _api.getJson('/api/v1/plugins');
      final provs = await _api.getJson('/api/v1/providers');

      if (!mounted) return;
      setState(() {
        _backendConnected = true;
        _doctor = doc;
        _lowMemoryMode = (doc['platform']?['low_memory_mode'] as bool?) ?? false;
        _projects = (projs['projects'] as List?) ?? const [];
        _models = (mdls['models'] as List?) ?? const [];
        _agents = (agts['agents'] as List?) ?? const [];
        _tasks = (tsks['tasks'] as List?) ?? const [];
        _experiments = (exps['experiments'] as List?) ?? const [];
        _mcpServers = (mcps['servers'] as List?) ?? const [];
        _plugins = (plugs['plugins'] as List?) ?? const [];
        _providers = (provs['providers'] as List?) ?? const [];

        if (_projects.isNotEmpty && _activeProjectId.isEmpty) {
          _activeProjectId = _projects.first['id'].toString();
        }
        if (_models.isNotEmpty && _inferModelIdCtrl.text.isEmpty) {
          _inferModelIdCtrl.text = _models.first['id'].toString();
        }
        _outputConsole = const JsonEncoder.withIndent('  ').convert(doc);
      });
      _connectWs();
    });
  }

  Future<String> _ensureProjectId() async {
    if (_activeProjectId.isNotEmpty) return _activeProjectId;
    final created = await _api.postJson('/api/v1/projects', {
      'name': _projectNameCtrl.text.trim().isEmpty
          ? 'Default Workspace'
          : _projectNameCtrl.text.trim(),
      'description': 'Created from Flutter Material 3 Studio',
    });
    _activeProjectId = created['id'].toString();
    return _activeProjectId;
  }

  Widget _buildJsonPanel(String title, String content) {
    return Card(
      margin: const EdgeInsets.only(top: 12),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            SelectableText(
              content,
              style: const TextStyle(fontFamily: 'monospace', fontSize: 12.5),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDashboard(AppStrings s) {
    final counts = (_doctor['counts'] as Map<String, dynamic>?) ?? const {};
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _buildQuickStart(),
        const SizedBox(height: 16),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: [
            _statCard('Проекты', '${counts['projects'] ?? _localDeviceProjects.length}', Icons.folder_outlined),
            _statCard('Модели', '${counts['models'] ?? _models.length}', Icons.hub_outlined),
            _statCard('Агенты', '${counts['agents'] ?? _agents.length}', Icons.smart_toy_outlined),
            _statCard('Эксперименты', '${counts['experiments'] ?? _experiments.length}', Icons.science_outlined),
          ],
        ),
        const SizedBox(height: 16),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(s.tr('create_project'), style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                TextField(
                  controller: _projectNameCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Название проекта',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 10,
                  children: [
                    FilledButton.icon(
                      icon: const Icon(Icons.add),
                      label: Text(s.tr('create_project')),
                      onPressed: _busy
                          ? null
                          : () => _runGuarded('Создание проекта', () async {
                                if (_backendConnected) {
                                  final created = await _api.postJson('/api/v1/projects', {
                                    'name': _projectNameCtrl.text.trim(),
                                    'description': 'Flutter Studio workspace',
                                  });
                                  _activeProjectId = created['id'].toString();
                                  await _refreshAll();
                                } else {
                                  setState(() {
                                    final localProj = {
                                      'id': 'local-${_localDeviceProjects.length + 1}',
                                      'name': _projectNameCtrl.text.trim(),
                                      'mode': 'device_offline_cache',
                                    };
                                    _localDeviceProjects.add(localProj);
                                    _outputConsole =
                                        'Проект сохранён локально на устройстве (Backend не подключён):\n${jsonEncode(localProj)}';
                                  });
                                }
                              }),
                    ),
                    OutlinedButton.icon(
                      icon: const Icon(Icons.sync),
                      label: const Text('Синхронизировать с Backend'),
                      onPressed: _busy ? null : _refreshAll,
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        if (_tasks.isNotEmpty)
          _buildJsonPanel(
            'Активные и последние задачи (${_tasks.length})',
            const JsonEncoder.withIndent('  ').convert(_tasks.take(5).toList()),
          ),
        if (_liveEvents.isNotEmpty)
          _buildJsonPanel('Поток событий WebSocket', _liveEvents.take(10).join('\n')),
      ],
    );
  }

  Widget _buildQuickStart() {
    final theme = Theme.of(context);
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text('Создайте свою модель', style: theme.textTheme.headlineSmall),
          const SizedBox(height: 8),
          Text('Расскажите, для чего она нужна, и добавьте текст. '
              'Параметры небольшой модели подберутся автоматически.',
              style: theme.textTheme.bodyMedium),
          const SizedBox(height: 16),
          if (!_backendConnected && !Platform.isAndroid) ...[
            Text('Вычислительное ядро не подключено. На Android этой версии '
                'обучение внутри приложения пока недоступно.',
                style: TextStyle(color: theme.colorScheme.error)),
            const SizedBox(height: 12),
          ],
          TextField(
            controller: _purposeCtrl, maxLength: 500, maxLines: 2,
            decoration: const InputDecoration(
              labelText: 'Для чего нужна модель?',
              hintText: 'Например, помощник по изучению языков',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _dataCtrl, maxLength: Platform.isAndroid ? 20000 : 100000, minLines: 4, maxLines: 8,
            decoration: const InputDecoration(
              labelText: 'Текст для обучения',
              hintText: 'Вставьте свои тексты, по одной записи на строку',
              border: OutlineInputBorder(),
            ),
          ),
          Text(Platform.isAndroid
              ? 'Используйте свои данные. Обучение проходит на устройстве без сети. Не закрывайте приложение.'
              : 'Используйте данные, на которые у вас есть права. Текст отправляется подключённому ядру.',
              style: theme.textTheme.bodySmall),
          const SizedBox(height: 14),
          Text('Примерный размер весов: ${_targetWeightMb.toStringAsFixed(1)} МиБ'),
          Slider(
            min: 0.1, max: 10, divisions: 99, value: _targetWeightMb,
            label: '${_targetWeightMb.toStringAsFixed(1)} МиБ',
            onChanged: _busy ? null : (v) => setState(() => _targetWeightMb = v),
          ),
          Text('Вес модели не равен потреблению памяти при обучении.',
              style: theme.textTheme.bodySmall),
          const SizedBox(height: 12),
          DropdownButtonFormField<int>(
            value: _loopPasses,
            decoration: const InputDecoration(
              labelText: 'Проходы трансформера', border: OutlineInputBorder(),
              helperText: 'Больше проходов требуют больше времени и памяти.',
              helperMaxLines: 2,
            ),
            items: List.generate(8, (i) => DropdownMenuItem(value: i + 1, child: Text('${i + 1}'))),
            onChanged: _busy ? null : (v) => setState(() => _loopPasses = v ?? 1),
          ),
          const SizedBox(height: 20),
          SizedBox(width: double.infinity, child: FilledButton.icon(
            icon: const Icon(Icons.auto_awesome),
            label: const Text('Создать и обучить'),
            onPressed: _busy || (!_backendConnected && !Platform.isAndroid) ? null : _startQuickTraining,
          )),
          if (_quickStatus.isNotEmpty) ...[
            const SizedBox(height: 12),
            SelectableText(_quickStatus, style: theme.textTheme.bodyMedium),
            if (!Platform.isAndroid) TextButton.icon(
              onPressed: _busy || Platform.isAndroid ? null : _refreshAll,
              icon: const Icon(Icons.refresh), label: const Text('Обновить задачи'),
            ),
          ],
          if (Platform.isAndroid) ...[
            const Divider(height: 32),
            Text('Проверить локальную модель', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            TextField(controller: _localPromptCtrl, maxLength: 5000,
              decoration: const InputDecoration(labelText: 'Начало текста', border: OutlineInputBorder())),
            OutlinedButton.icon(
              onPressed: _busy ? null : _generateLocal,
              icon: const Icon(Icons.chat_bubble_outline), label: const Text('Продолжить текст'),
            ),
            if (_localAnswer.isNotEmpty) SelectableText(_localAnswer),
          ],
        ]),
      ),
    );
  }

  Future<void> _generateLocal() async {
    await _runGuarded('Генерация на устройстве', () async {
      final response = await _offlineChannel.invokeMethod<String>('generate', jsonEncode({
        'prompt': _localPromptCtrl.text, 'max_chars': 64,
      }));
      final result = jsonDecode(response!) as Map<String, dynamic>;
      if (mounted) setState(() => _localAnswer = result['text'].toString());
    });
  }

  Future<void> _startQuickTraining() async {
    if (_purposeCtrl.text.trim().isEmpty || _dataCtrl.text.trim().isEmpty) {
      setState(() => _quickStatus = 'Заполните назначение модели и текст для обучения.');
      return;
    }
    await _runGuarded('Подготовка обучения', () async {
      if (Platform.isAndroid) {
        setState(() => _quickStatus = 'Обучение на телефоне. Не закрывайте приложение; это может занять до нескольких минут.');
        final response = await _offlineChannel.invokeMethod<String>('train', jsonEncode({
          'purpose': _purposeCtrl.text.trim(), 'text': _dataCtrl.text,
          'target_weight_mb': _targetWeightMb, 'loop_passes': _loopPasses,
        }));
        final metadata = jsonDecode(response!) as Map<String, dynamic>;
        if (mounted) setState(() => _quickStatus =
            'Обучение завершено на устройстве. Вес: ${metadata['weight_mb']} МиБ; '
            'шагов: ${metadata['steps']}. Модель очень маленькая и экспериментальная.');
        return;
      }
      final projectId = await _ensureProjectId();
      final created = await _api.postJson('/api/v1/models/automatic', {
        'project_id': projectId, 'purpose': _purposeCtrl.text.trim(),
        'text': _dataCtrl.text, 'target_weight_mb': _targetWeightMb,
        'loop_passes': _loopPasses,
      });
      final model = created['model'] as Map<String, dynamic>;
      final dataset = created['dataset'] as Map<String, dynamic>;
      final plan = created['plan'] as Map<String, dynamic>;
      _inferModelIdCtrl.text = model['id'].toString();
      // Expose the created model even if submitting the training task fails.
      if (mounted) setState(() => _quickStatus = 'Модель ${model['id']} создана. Запускаем обучение…');
      final task = await _api.postJson('/api/v1/training/run', {
        'model_id': model['id'], 'dataset_id': dataset['id'],
        'epochs': 4, 'batch_size': 1, 'max_seq_len': 64, 'wait': false,
      });
      if (mounted) setState(() => _quickStatus =
          'Задача ${task['id']} запущена. Приблизительный размер весов: '
          '${plan['estimated_weight_mb']} МиБ. Обновите задачи, чтобы увидеть результат.');
    });
  }

  Widget _statCard(String label, String value, IconData icon) {
    return SizedBox(
      width: 190,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Row(
            children: [
              Icon(icon, size: 28),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(label, style: Theme.of(context).textTheme.bodySmall),
                    Text(value, style: Theme.of(context).textTheme.headlineSmall),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildChat(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('AI Chat / Prompt Workspace', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                TextField(
                  controller: _chatCtrl,
                  maxLines: 3,
                  decoration: const InputDecoration(
                    labelText: 'Промпт (RU / TG / EN)',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                FilledButton.icon(
                  icon: const Icon(Icons.send),
                  label: const Text('Отправить'),
                  onPressed: _busy
                      ? null
                      : () => _runGuarded('AI Chat запрос', () async {
                            final pid = await _ensureProjectId();
                            final res = await _api.postJson('/api/v1/chat', {
                              'project_id': pid,
                              'message': _chatCtrl.text,
                            });
                            setState(() {
                              _outputConsole = const JsonEncoder.withIndent('  ').convert(res);
                            });
                          }),
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel('Ответ и RAG-цитаты', _outputConsole),
      ],
    );
  }

  Widget _buildModelStudio(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Model Builder (Создание модели по промпту)',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                TextField(
                  controller: _modelPromptCtrl,
                  maxLines: 3,
                  decoration: const InputDecoration(
                    labelText: 'Описание архитектуры и языков',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),
                DropdownButtonFormField<int>(
                  value: _loopPasses,
                  decoration: const InputDecoration(
                    labelText: 'Повторные проходы трансформера',
                    border: OutlineInputBorder(),
                    helperText: 'Больше проходов — больше времени и памяти, не гарантия лучшего ответа.',
                    helperMaxLines: 2,
                  ),
                  items: List.generate(8, (i) => DropdownMenuItem(
                    value: i + 1, child: Text('${i + 1}'),
                  )),
                  onChanged: _busy ? null : (value) => setState(() => _loopPasses = value ?? 1),
                ),
                const SizedBox(height: 10),
                FilledButton.icon(
                  icon: const Icon(Icons.auto_fix_high),
                  label: const Text('Создать модель и токенизатор'),
                  onPressed: _busy
                      ? null
                      : () => _runGuarded('Создание модели', () async {
                            final pid = await _ensureProjectId();
                            final mdl = await _api.postJson('/api/v1/models/from-prompt', {
                              'project_id': pid,
                              'prompt': _modelPromptCtrl.text,
                              'loop_passes': _loopPasses,
                            });
                            _inferModelIdCtrl.text = mdl['id'].toString();
                            await _refreshAll();
                            setState(() {
                              _outputConsole = const JsonEncoder.withIndent('  ').convert(mdl);
                            });
                          }),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Инференс и Экспорт (GGUF v3 / Safetensors)',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                TextField(
                  controller: _inferModelIdCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Model ID',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: _inferPromptCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Промпт для генерации',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 10,
                  runSpacing: 8,
                  children: [
                    FilledButton.icon(
                      icon: const Icon(Icons.bolt),
                      label: Text(s.tr('generate_text')),
                      onPressed: _busy
                          ? null
                          : () => _runGuarded('Генерация текста', () async {
                                final out = await _api.postJson('/api/v1/inference/generate', {
                                  'model_id': _inferModelIdCtrl.text.trim(),
                                  'prompt': _inferPromptCtrl.text,
                                  'max_tokens': 24,
                                });
                                setState(() {
                                  _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                                });
                              }),
                    ),
                    OutlinedButton(
                      onPressed: _busy
                          ? null
                          : () => _runGuarded('Экспорт GGUF v3', () async {
                                final id = _inferModelIdCtrl.text.trim();
                                final out = await _api.postJson('/api/v1/models/$id/export', {
                                  'format': 'gguf',
                                });
                                setState(() {
                                  _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                                });
                              }),
                      child: const Text('Экспорт GGUF v3'),
                    ),
                    OutlinedButton(
                      onPressed: _busy
                          ? null
                          : () => _runGuarded('Экспорт Safetensors', () async {
                                final id = _inferModelIdCtrl.text.trim();
                                final out = await _api.postJson('/api/v1/models/$id/export', {
                                  'format': 'safetensors',
                                });
                                setState(() {
                                  _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                                });
                              }),
                      child: const Text('Экспорт Safetensors'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel('Спецификация / Генерация / Экспорт', _outputConsole),
      ],
    );
  }

  Widget _buildAgentStudio(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Agent Builder & Runtime', style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                TextField(
                  controller: _agentPromptCtrl,
                  maxLines: 2,
                  decoration: const InputDecoration(
                    labelText: 'Описание агента и его прав',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: _agentTaskCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Ограниченная задача для запуска',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 10,
                  children: [
                    FilledButton(
                      onPressed: _busy
                          ? null
                          : () => _runGuarded('Создание и запуск агента', () async {
                                final pid = await _ensureProjectId();
                                final agt = await _api.postJson('/api/v1/agents/from-prompt', {
                                  'project_id': pid,
                                  'prompt': _agentPromptCtrl.text,
                                });
                                final runRes = await _api.postJson(
                                  '/api/v1/agents/${agt['id']}/run',
                                  {'prompt': _agentTaskCtrl.text},
                                );
                                await _refreshAll();
                                setState(() {
                                  _outputConsole =
                                      const JsonEncoder.withIndent('  ').convert(runRes);
                                });
                              }),
                      child: const Text('Создать и выполнить задачу'),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel('Агенты (${_agents.length}) и отчёт выполнения', _outputConsole),
      ],
    );
  }

  Widget _buildTrainingCenter(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        if (_lowMemoryMode)
          Card(
            color: Theme.of(context).colorScheme.secondaryContainer,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Text(s.tr('low_mem_notice')),
            ),
          ),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Training Center — Реальный цикл обучения и чекпоинты',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                TextField(
                  controller: _inferModelIdCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Model ID',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(value: 'scratch', label: Text('Scratch')),
                    ButtonSegment(value: 'finetune', label: Text('Fine-tune')),
                    ButtonSegment(value: 'transfer', label: Text('Transfer')),
                  ],
                  selected: {_trainMode},
                  onSelectionChanged: (sel) => setState(() => _trainMode = sel.first),
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _trainEpochsCtrl,
                        decoration: const InputDecoration(
                          labelText: 'Epochs',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: TextField(
                        controller: _trainLrCtrl,
                        decoration: const InputDecoration(
                          labelText: 'Learning Rate',
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                FilledButton.icon(
                  icon: const Icon(Icons.play_arrow),
                  label: Text(s.tr('run_training')),
                  onPressed: _busy
                      ? null
                      : () => _runGuarded('Обучение модели', () async {
                            final out = await _api.postJson('/api/v1/training/run', {
                              'model_id': _inferModelIdCtrl.text.trim(),
                              'mode': _trainMode,
                              'epochs': int.tryParse(_trainEpochsCtrl.text) ?? 4,
                              'learning_rate': double.tryParse(_trainLrCtrl.text) ?? 0.004,
                              'wait': true,
                            });
                            await _refreshAll();
                            setState(() {
                              _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                            });
                          }),
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel('Метрики обучения и SHA-256 чекпоинта', _outputConsole),
      ],
    );
  }

  Widget _buildResearchLab(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Research Lab & Experiment Registry',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                TextField(
                  controller: _researchQuestionCtrl,
                  maxLines: 2,
                  decoration: const InputDecoration(
                    labelText: 'Исследовательский вопрос',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                FilledButton.icon(
                  icon: const Icon(Icons.science),
                  label: const Text('Запустить контролируемый эксперимент'),
                  onPressed: _busy
                      ? null
                      : () => _runGuarded('Сравнительное исследование', () async {
                            final pid = await _ensureProjectId();
                            final out = await _api.postJson('/api/v1/research/study', {
                              'project_id': pid,
                              'question': _researchQuestionCtrl.text,
                              'baseline_arch': 'decoder_transformer',
                              'candidate_arch': 'hybrid_attention_ssm',
                              'epochs': 3,
                            });
                            await _refreshAll();
                            setState(() {
                              _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                            });
                          }),
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel('Факты vs Гипотезы и Experiment Registry', _outputConsole),
      ],
    );
  }

  Widget _buildExplorerAndDiff(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Prompt-to-Project (Правки промптом, Unified Diff, Тесты и Rollback)',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                TextField(
                  controller: _promptEditCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Инструкция изменения проекта',
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 10),
                Wrap(
                  spacing: 10,
                  children: [
                    FilledButton(
                      onPressed: _busy
                          ? null
                          : () => _runGuarded('Prompt-to-Project', () async {
                                final pid = await _ensureProjectId();
                                final out = await _api.postJson(
                                  '/api/v1/projects/$pid/prompt-edit',
                                  {'prompt': _promptEditCtrl.text},
                                );
                                _lastEditId = out['id'].toString();
                                setState(() {
                                  _outputConsole =
                                      'Edit ID: $_lastEditId\n\n${out['diff_text']}\n\nTests:\n${const JsonEncoder.withIndent('  ').convert(out['test_report'])}';
                                });
                              }),
                      child: Text(s.tr('apply_prompt_edit')),
                    ),
                    OutlinedButton(
                      onPressed: (_busy || _lastEditId.isEmpty)
                          ? null
                          : () => _runGuarded('Rollback правки', () async {
                                final out = await _api.postJson(
                                  '/api/v1/prompt-edits/$_lastEditId/rollback',
                                  {},
                                );
                                setState(() {
                                  _outputConsole =
                                      'Rollback выполнен:\n${const JsonEncoder.withIndent('  ').convert(out)}';
                                });
                              }),
                      child: Text(s.tr('rollback')),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel('Unified Diff и результат автотестов', _outputConsole),
      ],
    );
  }

  Widget _buildRegistry(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _registryQueryCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Поиск по моделям, агентам, датасетам и архитектурам',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                FilledButton(
                  onPressed: _busy
                      ? null
                      : () => _runGuarded('Поиск в реестре', () async {
                            final q = Uri.encodeQueryComponent(_registryQueryCtrl.text.trim());
                            final out = await _api.getJson('/api/v1/registry?q=$q');
                            setState(() {
                              _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                            });
                          }),
                  child: const Text('Найти'),
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel('Каталог моделей, архитектур, датасетов и экспериментов', _outputConsole),
      ],
    );
  }

  Widget _buildMcpAndPlugins(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [
                FilledButton(
                  onPressed: _busy
                      ? null
                      : () => _runGuarded('Проверка MCP STDIO', () async {
                            final probe =
                                await _api.postJson('/api/v1/mcp/mcp-builtin-local/probe', {});
                            final call = await _api.postJson(
                              '/api/v1/mcp/mcp-builtin-local/call',
                              {
                                'tool_name': 'analyze_corpus_stats',
                                'arguments': {'text': 'Салом ҷаҳон! Привет мир!'},
                              },
                            );
                            setState(() {
                              _outputConsole = const JsonEncoder.withIndent('  ')
                                  .convert({'probe': probe, 'call': call});
                            });
                          }),
                  child: const Text('Проверить встроенный MCP сервер'),
                ),
                OutlinedButton(
                  onPressed: _busy
                      ? null
                      : () => _runGuarded('Запуск плагина Tajik Inspector', () async {
                            final out = await _api.postJson(
                              '/api/v1/plugins/tajik_text_inspector/run',
                              {
                                'inputs': {'text': 'Салом ҷаҳон! Забони тоҷикӣ: ғӣқӯҳҷ.'}
                              },
                            );
                            setState(() {
                              _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                            });
                          }),
                  child: const Text('Запустить плагин Tajik Inspector'),
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel(
          'MCP серверы (${_mcpServers.length}) и Плагины (${_plugins.length})',
          _outputConsole,
        ),
      ],
    );
  }

  Widget _buildProviders(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('LLM Провайдеры и Секреты (Без скрытого переключения)',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 8),
                const Text(
                  'Политика маршрутизации: явный выбор модели. Переключение между локальной и внешней платной моделью никогда не выполняется скрыто.',
                ),
                const SizedBox(height: 10),
                FilledButton(
                  onPressed: _busy
                      ? null
                      : () => _runGuarded('Проверка локального провайдера', () async {
                            final out =
                                await _api.postJson('/api/v1/providers/provider-local/test', {});
                            setState(() {
                              _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                            });
                          }),
                  child: const Text('Проверить Local Provider'),
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel(
          'Зарегистрированные провайдеры (${_providers.length})',
          const JsonEncoder.withIndent('  ').convert(_providers),
        ),
      ],
    );
  }

  Widget _buildSystemMonitor(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('System Monitor & Platform Delivery',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 8),
                Text(_backendProc.describePlatformStrategy()),
                const SizedBox(height: 10),
                FilledButton.icon(
                  icon: const Icon(Icons.health_and_safety),
                  label: const Text('Запустить полную диагностику (doctor)'),
                  onPressed: _busy ? null : _refreshAll,
                ),
              ],
            ),
          ),
        ),
        _buildJsonPanel('Отчёт диагностики ядра', _outputConsole),
      ],
    );
  }

  Widget _buildSettings(AppStrings s) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Режим работы (Local / Remote / Hybrid / Offline)',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(value: 'local', label: Text('Local')),
                    ButtonSegment(value: 'remote', label: Text('Remote')),
                    ButtonSegment(value: 'hybrid', label: Text('Hybrid')),
                    ButtonSegment(value: 'offline', label: Text('Offline')),
                  ],
                  selected: {_api.mode},
                  onSelectionChanged: (sel) async {
                    final chosen = sel.first;
                    setState(() => _api.mode = chosen);
                    if (_backendConnected) {
                      await _runGuarded('Переключение режима ($chosen)', () async {
                        final out = await _api.postJson('/api/v1/mode', {
                          'mode': chosen,
                          'remote_server_url': _serverUrlCtrl.text.trim(),
                        });
                        setState(() {
                          _outputConsole = const JsonEncoder.withIndent('  ').convert(out);
                        });
                      });
                    }
                  },
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _serverUrlCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Backend / Remote Core URL',
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (val) => setState(() => _api.baseUrl = val.trim()),
                ),
                const SizedBox(height: 10),
                FilledButton(
                  onPressed: () {
                    setState(() => _api.baseUrl = _serverUrlCtrl.text.trim());
                    _refreshAll();
                  },
                  child: const Text('Применить адрес и проверить соединение'),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 12),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Тема, Язык (RU / TG / EN) и Доступность',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 10),
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(value: 'ru', label: Text('Русский')),
                    ButtonSegment(value: 'tg', label: Text('Тоҷикӣ')),
                    ButtonSegment(value: 'en', label: Text('English')),
                  ],
                  selected: {widget.localeCode},
                  onSelectionChanged: (sel) =>
                      widget.onSettingsChanged(localeCode: sel.first),
                ),
                const SizedBox(height: 12),
                SegmentedButton<ThemeMode>(
                  segments: const [
                    ButtonSegment(value: ThemeMode.dark, label: Text('Dark')),
                    ButtonSegment(value: ThemeMode.light, label: Text('Light')),
                    ButtonSegment(value: ThemeMode.system, label: Text('System')),
                  ],
                  selected: {widget.themeMode},
                  onSelectionChanged: (sel) =>
                      widget.onSettingsChanged(themeMode: sel.first),
                ),
                const SizedBox(height: 12),
                Text('Масштаб текста: ${widget.textScale.toStringAsFixed(2)}x'),
                Slider(
                  min: 0.85,
                  max: 1.4,
                  divisions: 11,
                  value: widget.textScale,
                  label: '${widget.textScale.toStringAsFixed(2)}x',
                  onChanged: (v) => widget.onSettingsChanged(textScale: v),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildActivePage(AppStrings s) {
    switch (_selectedIndex) {
      case 0:
        return _buildDashboard(s);
      case 1:
        return _buildChat(s);
      case 2:
        return _buildModelStudio(s);
      case 3:
        return _buildAgentStudio(s);
      case 4:
        return _buildTrainingCenter(s);
      case 5:
        return _buildResearchLab(s);
      case 6:
        return _buildExplorerAndDiff(s);
      case 7:
        return _buildRegistry(s);
      case 8:
        return _buildMcpAndPlugins(s);
      case 9:
        return _buildProviders(s);
      case 10:
        return _buildSystemMonitor(s);
      case 11:
        return _buildSettings(s);
      default:
        return _buildDashboard(s);
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.strings;
    final width = MediaQuery.of(context).size.width;
    final isWide = width >= 820;

    return Scaffold(
      appBar: AppBar(
        title: Text(s.tr('app_title')),
        actions: [
          if (_busy)
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 12),
              child: SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Chip(
              label: Text(
                Platform.isAndroid && !_backendConnected ? 'НА УСТРОЙСТВЕ' :
                    '${_api.mode.toUpperCase()}${_backendConnected ? " • ONLINE" : " • НЕТ СВЯЗИ"}',
              ),
            ),
          ),
        ],
      ),
      drawer: isWide
          ? null
          : Drawer(
              child: SafeArea(
                child: ListView(
                  children: [
                    for (var i = 0; i < kSections.length; i++)
                      ListTile(
                        leading: Icon(kSections[i].icon),
                        title: Text(s.tr(kSections[i].keyName)),
                        selected: _selectedIndex == i,
                        onTap: () {
                          setState(() => _selectedIndex = i);
                          Navigator.of(context).pop();
                        },
                      ),
                  ],
                ),
              ),
            ),
      body: Column(
        children: [
          if (_api.mode == 'offline')
            MaterialBanner(
              content: Text(s.tr('offline_notice')),
              actions: [
                TextButton(
                  onPressed: () => setState(() => _api.mode = 'local'),
                  child: const Text('Switch to Local'),
                ),
              ],
            ),
          if (_statusBanner.isNotEmpty)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              color: Theme.of(context).colorScheme.surfaceContainerHighest,
              child: Text(_statusBanner, style: Theme.of(context).textTheme.bodySmall),
            ),
          Expanded(
            child: Row(
              children: [
                if (isWide)
                  SingleChildScrollView(
                    child: IntrinsicHeight(
                      child: NavigationRail(
                        selectedIndex: _selectedIndex,
                        onDestinationSelected: (idx) => setState(() => _selectedIndex = idx),
                        labelType: NavigationRailLabelType.all,
                        destinations: [
                          for (final sec in kSections)
                            NavigationRailDestination(
                              icon: Icon(sec.icon),
                              label: Text(s.tr(sec.keyName)),
                            ),
                        ],
                      ),
                    ),
                  ),
                if (isWide) const VerticalDivider(width: 1),
                Expanded(child: _buildActivePage(s)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
