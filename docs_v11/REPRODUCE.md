# REPRODUCE.md — v0.11

```bash
pip install pytest hypothesis numpy scikit-learn   # dev-зависимости (pip в песочнице не персистит)
python -m pytest tests_v11 -q                       # 56 passed
python bench_v11/gen_corpus_v11.py                  # перегенерация корпуса (детерминировано)
python bench_v11/eval_intent.py                     # → results_v11/intent_eval.json
python -m rasm_cim_v11.cli language-check "Когда полетел Гагарин в космос?"
python -m rasm_cim_v11.cli ask-text "Какая столица у Франции?"
python -m pytest test_rasm*.py rasm_cim_v8/tests tests_v9 tests_v10 -q   # исторический gate
md5sum -c results_v11/freeze_manifest_baseline_v0.10.md5 | grep -cv ': OK'   # → 0
```
Дата прогона, среда и пин-числа: results_v11/{test_run.log,intent_eval.json,bench_latency.json}.
