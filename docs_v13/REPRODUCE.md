# REPRODUCE.md — v0.13

```bash
python -m pytest tests_v13 -q                       # 40 passed
rm -rf /tmp/kb13s /tmp/ctx13s
python -m rasm_cim_v12.cli seed-demo /tmp/kb13s     # offline демо-факты
python -m rasm_cim_v13.cli ask /tmp/ctx13s --kb /tmp/kb13s "Какая столица у Франции?"
python -m rasm_cim_v13.cli ask /tmp/ctx13s --kb /tmp/kb13s "А её население?"
python -m rasm_cim_v13.cli ask /tmp/ctx13s --kb /tmp/kb13s "Что такое киты? И кто написал Войну и мир?"
python -m rasm_cim_v13.cli ask /tmp/ctx13s --kb /tmp/kb13s "Какие столицы у Германии и Франции?"
python -m rasm_cim_v13.cli ask /tmp/ctx13s --kb /tmp/kb13s "А её?"      # exit 4 + CLARIFY
python -m rasm_cim_v13.cli show-context /tmp/ctx13s
python -m rasm_cim_v13.cli export-session /tmp/ctx13s session.json
python -m rasm_cim_v13.cli reset-context /tmp/ctx13s
md5sum -c results_v11/freeze_manifest_baseline_v0.10.md5 | grep -cv ': OK'   # → 0
```
Приватность: context.json хранит сырьё реплик в открытом виде — держите каталог в приватной
директории (на устройстве: ~/.rasm-cim с chmod 700); внешние подключения отсутствуют.
Логи: results_v13/{test_run.log,bench_dialogue.json}.
