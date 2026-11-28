# Android ARM64 & Termux Runtime Architecture

## Важное архитектурное правило
Flutter APK **не** запускает обычный CPython внутри Dart VM напрямую. На Android ModelForge AI поддерживает три честных и проверенных режима:

1. **Termux Local Runtime (без root, на том же смартфоне)**:
   - Установите Termux (F-Droid, ARM64) и выполните `bash platform/termux/install_termux.sh`.
   - Запустите локальный loopback-сервис `bash platform/termux/start_modelforge_termux.sh` (`127.0.0.1:8765`).
   - В приложении `ModelForge-Android-ARM64.apk` в режиме `Local Mode` адрес `http://127.0.0.1:8765` подключается к локальному ядру без выхода в интернет.
   - На смартфонах с `<= 3.5 ГБ RAM` автоматически активируется **Low Memory Mode** (`max_params <= 2.5M`, `batch_size <= 4`, `seq_len <= 128`).
2. **Remote / Hybrid Mode**:
   - В разделе `Settings` укажите URL вашего настольного ПК или сервера (`http://<IP>:8765`), где запущен `modelforge server`.
3. **Offline On-Device Cache**:
   - При отсутствии сети и выключенном Termux приложение открывается без зависаний, позволяет просматривать локальные заметки проектов и явно помечает сетевые/вычислительные вызовы как недоступные.
