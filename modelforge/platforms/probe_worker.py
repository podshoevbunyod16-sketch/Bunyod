"""Isolated CPU probe. Not a full training run."""
from __future__ import annotations
import hashlib
import json
import sys
import time


def run(payload):
    import numpy as np
    rng = np.random.default_rng(42)
    if payload['kind'] == 'matmul':
        n, repeats = payload['size'], payload['repeats']
        a = rng.standard_normal((n, n), dtype=np.float32)
        b = rng.standard_normal((n, n), dtype=np.float32)
        a @ b
        start = time.perf_counter()
        for _ in range(repeats):
            a @ b
        seconds = time.perf_counter() - start
        return {'size': n, 'repeats': repeats, 'dtype': 'float32', 'seconds': seconds,
                'gflops': 2 * n ** 3 * repeats / seconds / 1e9, 'source': 'timed numpy.matmul; single BLAS thread'}
    from modelforge.architectures.models import build_model_from_spec
    from modelforge.training.engine import AdamWOptimizer
    spec = payload['spec']
    model = build_model_from_spec(spec, seed=42)
    optimizer = AdamWOptimizer(model, lr=0.003)
    x = rng.integers(1, spec['vocab_size'], size=(payload['batch_size'], spec['context_length']))
    y = rng.integers(1, spec['vocab_size'], size=x.shape)
    before = hashlib.sha256(model.params['lm_head'].data.tobytes()).hexdigest()
    start = time.perf_counter()
    optimizer.zero_grad()
    loss = model.compute_loss(x, y, training=True)
    loss.backward()
    optimizer.clip_grad_norm(1.0)
    optimizer.step()
    elapsed = time.perf_counter() - start
    after = hashlib.sha256(model.params['lm_head'].data.tobytes()).hexdigest()
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = int(rss if sys.platform == 'darwin' else rss * 1024)
        peak_source = 'resource.getrusage(RUSAGE_SELF).ru_maxrss'
    except ImportError:
        peak = None
        peak_source = 'parent psutil sampling'
    return {'spec': spec, 'batch_size': payload['batch_size'], 'parameters': model.parameter_count(),
            'step_seconds': elapsed, 'loss': float(loss.data), 'weights_changed': before != after,
            'peak_rss_bytes_worker': peak, 'peak_source': peak_source}


if __name__ == '__main__':
    print(json.dumps(run(json.loads(sys.argv[1]))))
