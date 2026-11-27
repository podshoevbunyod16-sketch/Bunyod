# REPRODUCE.md — v0.12

```bash
python -m pytest tests_v12 -q                                  # 27 passed
rm -rf /tmp/kbdemo && python -m rasm_cim_v12.cli seed-demo /tmp/kbdemo
python -m rasm_cim_v12.cli kb-info /tmp/kbdemo
python -m rasm_cim_v12.cli conflicts /tmp/kbdemo                # Дракон → polarity_conflict
python -m rasm_cim_v12.cli kb-query /tmp/kbdemo --subject Таджикистан
python -m rasm_cim_v12.cli export-jsonl /tmp/kbdemo /tmp/kbdemo.jsonl
python -m rasm_cim_v12.cli import-jsonl /tmp/kbdemo2 /tmp/kbdemo.jsonl   # re-import OK
python -m rasm_cim_v12.cli import-jsonl /tmp/kbdemo2 /tmp/kbdemo.jsonl   # imported=0
python -m pytest tests_v11 tests_v12 -q                         # 83 passed
md5sum -c results_v11/freeze_manifest_baseline_v0.10.md5 | grep -cv ': OK'   # → 0
```
Логи: results_v12/{test_run.log,bench_kb.json}.
