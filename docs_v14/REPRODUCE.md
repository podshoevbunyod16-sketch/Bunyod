# REPRODUCE.md — v0.14

```bash
python -m pytest tests_v14 -q                                  # 21 passed
python bench_v14/bench_reasoning.py                            # latency/RSS/quality
```

Interactive spot-check (any project KB):

```bash
python -c "
import sys; sys.path.insert(0,'.')
from rasm_cim_v12.knowledge import KnowledgeStore
from rasm_cim_v14.reasoning import ask
kb = KnowledgeStore.init('/tmp/kb-check')
print(ask(kb, 'Какая столица у Франции?')['answer'])
print(ask(kb, 'Что такое Киты?', max_depth=3)['answer'])
print(ask(kb, 'Где расположен Париж?', max_depth=3)['answer'])
"
```

Freeze check: `md5sum -c results_v11/freeze_manifest_baseline_v0.10.md5 | grep -cv ': OK'` → 0.

Logs: `results_v14/test_run.log`, `results_v14/bench_reasoning.json`.
