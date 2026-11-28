# Руководство контрибьютора (Contributing to ModelForge AI)

## Правила разработки

1. **Никаких заглушек под видом готовых функций**: любая опциональная или ещё не реализованная возможность должна явно помечаться (`NOT_IMPLEMENTED` / `optional_not_configured`) и честно отражаться в `docs/KNOWN_LIMITATIONS.md`.
2. **Обязательные реальные проверки**:
   - Изменения в тензорном движке или архитектурах проверяются численным `gradcheck` (`tests/test_autograd_and_architectures.py`).
   - Изменения в обучении обязаны проверять `weights_changed == True` и снижение функции потерь на контрольном корпусе (`tests/test_training_inference_export.py`).
   - Изменения в `Flutter` проверяются `flutter analyze` и `flutter test`.
3. **Безопасность по умолчанию**:
   - Не храните ключи API, `.jks`/`.keystore` и пароли в Git.
   - Не ослабляйте `resolve_within_workspace` и `sanitize_untrusted_content`.
   - Разрушительные, платные и внешние действия должны проходить через `ConfirmationManager`.

## Локальный запуск проверок перед созданием PR

```bash
python -m pytest -v
python scripts/smoke_e2e.py
```
