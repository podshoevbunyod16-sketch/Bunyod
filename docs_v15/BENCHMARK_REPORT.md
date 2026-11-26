# BENCHMARK_REPORT.md — v0.15 (host, Linux x86-64, Python 3.11.2)

Источник: results_v15/eval_baseline_vs_learned.log + results_v15/eval_summary.json.
Метод: 30 повторов/вопрос×8 KB-answerable, фиксированная demo KB (seed-demo).
Устройство (Termux) НЕ измерялось.

| сплит | режим | intent_acc | verdict_acc (n) | unsup.rate | p50 ms | RSS MB | reg. |
|---|---|---|---|---|---|---|---|
| dev(n=84) | baseline | 0.8333 | 1.0 (29) | 0.0 | 0.67 | 18.1 | 6/6 |
| dev | learned lv_000 | **0.8571** | 1.0 (29) | 0.0 | 0.67 | 18.4 | 6/6 |
| test(n=88) | baseline | 0.7614 | 1.0 (32) | 0.0 | 0.66 | 18.1 | 6/6 |
| test | learned lv_000 | **0.7841** | 1.0 (32) | 0.0 | 0.65 | 18.3 | 6/6 |

Дельта intent_acc: dev +2.38pp, test +2.27pp — маргинальная, без significance-теста,
fallback срабатываний 2/84 и 4/88. Мы НЕ утверждаем улучшения качества понимания;
фиксируем отсутствие деградации (p50-задержка и RSS unchanged, regression 6/6).

Ablation (attribution): вся измеримая дельта — вклад intent_fallback; predicate_hints и
alias_delta в lv_000 не применялись (0 коррекций свёрнуто). Удаление fallback ⇒ dev/test
возвращаются ровно к 0.8333/0.7614 (eval --learned выключен по умолчанию в eval без флага).

Training cost: построение артефакта < 1 s CPU (265 train), размер файла ~150 KB.
