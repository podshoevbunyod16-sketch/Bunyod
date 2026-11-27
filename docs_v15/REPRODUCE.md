# REPRODUCE.md — v0.15

```bash
# corpus (deterministic, committed; rebuild = same sha256)
python -m rasm_cim_v15.cli corpus-info
python -m rasm_cim_v15.cli build-corpus        # rewrites corpus/*.jsonl + manifest.json

# baseline & learned eval
rm -rf /tmp/kb15r && python -m rasm_cim_v12.cli seed-demo /tmp/kb15r
python -m rasm_cim_v15.cli eval --kb /tmp/kb15r --split dev
python -m rasm_cim_v15.cli eval --kb /tmp/kb15r --split dev --learned
python -m rasm_cim_v15.cli eval --kb /tmp/kb15r --split test --learned

# train & rollback
python -m rasm_cim_v15.cli train               # creates lv_NNN (parent = current)
python -m rasm_cim_v15.cli corrections          # list events + chain verification
python -m rasm_cim_v15.cli rollback lv_000

# correction e2e
python -m rasm_cim_v15.cli ask --kb /tmp/kb15r "Столицу Франции зовут?"
python -m rasm_cim_v15.cli correct --kb /tmp/kb15r --kind relation \
    --wrong "" --correct capital_of --target-text "Столицу Франции зовут?" --apply
python -m rasm_cim_v15.cli ask --kb /tmp/kb15r "Столицу Франции зовут?"   # → Париж [st_…]
python -m rasm_cim_v15.cli rollback lv_000                                # поведение вернулось

python -m pytest tests_v15 -q            # 14 passed
md5sum -c results_v11/freeze_manifest_baseline_v0.10.md5 | grep -cv ': OK'   # 0
```
Логи: results_v15/{test_run.log, eval_baseline_vs_learned.log, eval_summary.json}.
