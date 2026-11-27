# Установка и запуск ModelForge AI по платформам

## 1. Linux x64

### Вариант А — Готовые сборки (`AppImage` / `DEB`)
- **AppImage**:
  ```bash
  chmod +x ModelForge-Linux-x64.AppImage
  ./ModelForge-Linux-x64.AppImage
  ```
- **Ubuntu / Debian (`.deb`)**:
  ```bash
  sudo apt-get update && sudo apt-get install -y ./ModelForge-Linux-x64.deb
  modelforge-ai        # запуск Flutter Desktop GUI (со встроенным локальным backend)
  modelforge doctor    # запуск CLI
  ```

### Вариант Б — Из исходного кода
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[full,dev]"
modelforge doctor
modelforge server --host 0.0.0.0 --port 8765
```

---

## 2. Windows x64

- **Установщик (`ModelForge-Windows-x64-Setup.exe`)**: запустите установщик (не требует прав администратора, устанавливается в профиль пользователя вместе со встроенным `modelforge-backend.exe`), затем откройте **ModelForge AI** из меню Пуск.
- **Portable-архив (`ModelForge-Windows-x64-Portable.zip`)**: распакуйте архив в любую папку и запустите `modelforge_app.exe` (GUI автоматически поднимает локальный `backend\modelforge-backend.exe` на `127.0.0.1:8765`).
- **Удаление**: через стандартный раздел Windows «Приложения и возможности» или ярлык `Uninstall ModelForge AI`.

---

## 3. Android ARM64 и Termux ARM64 (без root)

> **Важно**: Flutter APK не запускает CPython внутри Dart VM напрямую. Для локальных вычислений на смартфоне используется безопасная связка с локальным сервисом Termux (`127.0.0.1:8765`) либо подключение к настольному серверу в `Remote / Hybrid Mode`.

### Шаг 1 — Установка Android GUI
Установите `ModelForge-Android-ARM64.apk` на смартфон с архитектурой `arm64-v8a` (Android 10+).  
*Примечание по подписи*: если в репозитории не заданы секреты релизного ключа (`ANDROID_KEYSTORE_*`), APK в артефактах CI подписан тестовым отладочным ключом (см. `ModelForge-Android-ARM64.SIGNING_STATUS.txt`); при установке разрешите установку из вашего файлового менеджера.

### Шаг 2 — Локальное ядро и CLI в Termux ARM64 (без root)
Установите Termux (из F-Droid) и выполните:
```bash
git clone https://github.com/podshoevbunyod16-sketch/Bunyod.git ModelForge-AI
cd ModelForge-AI
bash platform/termux/install_termux.sh
bash platform/termux/start_modelforge_termux.sh
```
Скрипт автоматически включает `Low Memory Mode` для смартфонов с 3–4 ГБ RAM и поднимает локальный сервер только на `127.0.0.1:8765`. После этого Android-приложение ModelForge AI в режиме `Local` сразу подключается к локальному ядру.

---

## 4. macOS (Apple Silicon & Intel)

Откройте `ModelForge-macOS.dmg` и перетащите `ModelForge AI.app` в папку `Applications`.  
*Примечание*: при отсутствии платного сертификата Apple Developer ID и нотаризации сборка не подписана Apple (`ad-hoc`); при первом запуске откройте приложение через контекстное меню `Открыть` (`Right-Click -> Open`) или выполните `xattr -cr "/Applications/ModelForge AI.app"`.
