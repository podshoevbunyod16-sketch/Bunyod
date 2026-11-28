"""Reproducible stage-0 experiment; never train or evaluate on the test split."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'experiments/stage0/data'
CONFIG = ROOT / 'experiments/stage0/config.json'
OUT = ROOT / 'experiments/stage0/runs/baseline_seed42'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise SystemExit(f'{name}=1 is required for reproducibility')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise SystemExit('Commit the experiment and dataset before running; worktree must be clean')
    if OUT.exists():
        raise SystemExit('Run directory already exists: do not overwrite fixed results')
    import numpy as np
    from modelforge.architectures.models import build_model_from_spec
    from modelforge.mobile.offline import train, generate
    from modelforge.tokenizers.tokenizer import CharTokenizer

    config = json.loads(CONFIG.read_text(encoding='utf8'))
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))
    for name, expected in manifest['files'].items():
        if sha(DATA / name) != expected:
            raise SystemExit(f'Frozen data hash mismatch: {name}')
    split = json.loads((DATA / 'split.json').read_text(encoding='utf8'))
    if split['seed'] != config['split_seed']:
        raise SystemExit('Unexpected split seed')
    texts = {lang: (DATA / f'{lang}.txt').read_text(encoding='utf8').splitlines() for lang in ('ru', 'tg')}
    groups: dict[str, dict[str, list[str]]] = {}
    seen: set[str] = set()
    for part in ('train', 'validation', 'test'):
        groups[part] = {}
        for lang in ('ru', 'tg'):
            chosen = [texts[lang][i] for i in split['splits'][part][lang]]
            for row in chosen:
                normalized = unicodedata.normalize('NFC', ' '.join(row.casefold().split()))
                if not normalized or normalized in seen:
                    raise SystemExit('Empty or duplicate record across splits')
                seen.add(normalized)
            groups[part][lang] = chosen
    if len(seen) != sum(len(rows) for rows in texts.values()):
        raise SystemExit('Split is incomplete or repeats records')
    prompts = json.loads((DATA / 'prompts.json').read_text(encoding='utf8'))
    if len(prompts) != 10 or sum(x['lang'] == 'ru' for x in prompts) != 5 or sum(x['lang'] == 'tg' for x in prompts) != 5:
        raise SystemExit('Invalid fixed prompt set')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    OUT.mkdir(parents=True)
    meta = {'code_commit': commit, 'script_sha256': sha(Path(__file__)),
            'config_sha256': sha(CONFIG), 'dataset_manifest_sha256': sha(DATA / 'manifest.json'),
            'dataset_files': manifest['files'], 'config': config,
            'command': 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/stage0_baseline.py',
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'numpy': np.__version__, 'thread_env': {k: os.environ[k] for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    (OUT / 'environment.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    train_text = '\n'.join(groups['train']['ru'] + groups['train']['tg'])
    started = time.monotonic()
    try:
        training = train(str(OUT), 'synthetic baseline', train_text,
                         target_mb=config['target_weight_mb'], passes=config['loop_passes'],
                         available_mb=config['available_mb_guard'], max_seconds=config['max_seconds'])
        with np.load(OUT / 'offline_model.npz', allow_pickle=False) as data:
            saved = json.loads(str(data['metadata']))
            tokenizer = CharTokenizer(saved['vocabulary'])
            model = build_model_from_spec(saved['spec'], seed=config['model_seed'])
            model.load_weights_dict({key: data[key] for key in data.files if key != 'metadata'})
        scores = {}
        for lang in ('ru', 'tg'):
            nll = 0.0
            count = 0
            for record in groups['validation'][lang]:
                ids = tokenizer.encode(record)
                if len(ids) != len(record) or tokenizer.unk_id in ids:
                    raise ValueError(f'Validation {lang} includes unsupported characters; cannot claim bpc')
                history = [tokenizer.bos_id]
                for target in ids:
                    context = np.array([history[-model.context_length:]], dtype=np.int64)
                    logits = model.forward(context, training=False).data[0, -1].astype(np.float64)
                    maximum = float(np.max(logits))
                    nll += maximum + math.log(float(np.exp(logits - maximum).sum())) - float(logits[target])
                    count += 1
                    history.append(target)
            scores[lang] = {'records': len(groups['validation'][lang]), 'characters': count,
                            'nll_nats': nll, 'loss_nats_per_character': nll/count,
                            'bits_per_character': nll/count/math.log(2)}
        combined_nll = sum(v['nll_nats'] for v in scores.values())
        combined_count = sum(v['characters'] for v in scores.values())
        scores['combined'] = {'characters': combined_count, 'nll_nats': combined_nll,
                              'loss_nats_per_character': combined_nll/combined_count,
                              'bits_per_character': combined_nll/combined_count/math.log(2)}
        generations = [{**p, 'generated_text': generate(str(OUT), p['prompt'], config['generation_max_chars'])} for p in prompts]
        results = {'status': 'completed', 'training': training, 'validation': scores,
                   'parameter_count': model.parameter_count(), 'checkpoint_sha256': sha(OUT/'offline_model.npz'),
                   'elapsed_total_seconds': time.monotonic()-started,
                   'completed_utc': datetime.now(timezone.utc).isoformat()}
        (OUT/'generation.json').write_text(json.dumps(generations,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except Exception as exc:
        (OUT/'failure.json').write_text(json.dumps({'error':repr(exc),'elapsed_seconds':time.monotonic()-started},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        raise


if __name__ == '__main__':
    main()
