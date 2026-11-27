/// Multilingual UI strings (Russian, Tajik Cyrillic, English) for ModelForge AI.
library;

class AppStrings {
  final String localeCode;
  const AppStrings(this.localeCode);

  static const Map<String, Map<String, String>> _table = {
    'ru': {
      'app_title': 'ModelForge AI',
      'dashboard': 'Панель',
      'chat': 'ИИ Чат',
      'model_studio': 'Модели',
      'agent_studio': 'Агенты',
      'training': 'Обучение',
      'research': 'Исследования',
      'explorer': 'Проект и Diff',
      'registry': 'Реестр',
      'mcp_plugins': 'MCP и Плагины',
      'providers': 'Провайдеры',
      'monitor': 'Мониторинг',
      'settings': 'Настройки',
      'create_project': 'Создать проект',
      'run_training': 'Запустить обучение',
      'generate_text': 'Сгенерировать текст',
      'apply_prompt_edit': 'Применить промпт-правку',
      'rollback': 'Откатить (Rollback)',
      'offline_notice': 'Offline Mode активен: сетевые вызовы заблокированы.',
      'low_mem_notice': 'Low Memory Mode (<= 3.5 ГБ RAM): безопасные лимиты включены.',
    },
    'tg': {
      'app_title': 'ModelForge AI',
      'dashboard': 'Лавҳаи асосӣ',
      'chat': 'Гуфтугӯи ЗС',
      'model_studio': 'Моделҳо',
      'agent_studio': 'Агентҳо',
      'training': 'Омӯзиш',
      'research': 'Тадқиқот',
      'explorer': 'Лоиҳа ва Diff',
      'registry': 'Феҳрист',
      'mcp_plugins': 'MCP ва Плагинҳо',
      'providers': 'Провайдерҳо',
      'monitor': 'Назорат',
      'settings': 'Танзимот',
      'create_project': 'Сохтани лоиҳа',
      'run_training': 'Оғози омӯзиш',
      'generate_text': 'Тавлиди матн',
      'apply_prompt_edit': 'Татбиқи тағйирот бо промпт',
      'rollback': 'Бозгашт (Rollback)',
      'offline_notice': 'Ҳолати офлайн фаъол аст: шабака хомӯш аст.',
      'low_mem_notice': 'Ҳолати хотираи кам (<= 3.5 ГБ RAM) фаъол аст.',
    },
    'en': {
      'app_title': 'ModelForge AI',
      'dashboard': 'Dashboard',
      'chat': 'AI Chat',
      'model_studio': 'Model Studio',
      'agent_studio': 'Agent Studio',
      'training': 'Training Center',
      'research': 'Research Lab',
      'explorer': 'Explorer & Diff',
      'registry': 'Registry',
      'mcp_plugins': 'MCP & Plugins',
      'providers': 'Providers',
      'monitor': 'System Monitor',
      'settings': 'Settings',
      'create_project': 'Create Project',
      'run_training': 'Start Training',
      'generate_text': 'Generate Text',
      'apply_prompt_edit': 'Apply Prompt Edit',
      'rollback': 'Rollback',
      'offline_notice': 'Offline Mode active: network operations are disabled.',
      'low_mem_notice': 'Low Memory Mode (<= 3.5 GB RAM): safe caps active.',
    },
  };

  String tr(String key) {
    final langMap = _table[localeCode] ?? _table['ru']!;
    return langMap[key] ?? _table['en']![key] ?? key;
  }
}
