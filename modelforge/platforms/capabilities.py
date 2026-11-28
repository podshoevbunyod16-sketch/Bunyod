"""Measured device capabilities; estimates are never treated as measurements."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


@dataclass(frozen=True)
class ProbeConfig:
    reserve_fraction: float = 0.35
    total_timeout_seconds: float = 55.0
    per_probe_seconds: float = 10.0
    batch_size: int = 1
    context_length: int = 16
    vocab_size: int = 256
    steps_per_epoch: int = 100
    candidates: tuple[int, ...] = (500_000, 1_000_000, 3_000_000, 8_000_000, 15_000_000, 30_000_000)

    def __post_init__(self):
        if not 0 <= self.reserve_fraction < 1 or not 0 < self.total_timeout_seconds <= 60 or not 0 < self.per_probe_seconds <= 60:
            raise ValueError('Invalid reserve or probe timeout')
        if not 1 <= self.batch_size <= 16 or not 1 <= self.context_length <= 256 or not 32 <= self.vocab_size <= 4096 or not self.steps_per_epoch > 0:
            raise ValueError('Invalid benchmark dimensions')
        if not self.candidates or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.candidates) or tuple(sorted(set(self.candidates))) != self.candidates:
            raise ValueError('Candidates must be positive, unique and increasing')


def memory_budget(available_bytes: int, reserve_fraction: float) -> int:
    if available_bytes < 0 or not 0 <= reserve_fraction < 1:
        raise ValueError('Invalid budget inputs')
    return int(available_bytes * (1 - reserve_fraction))


def _cgroup_limit():
    """Read the effective v2 cgroup and ancestors; never assume root is our group."""
    root = Path('/sys/fs/cgroup')
    try:
        lines = Path('/proc/self/cgroup').read_text().splitlines()
        relative = next(line.split(':', 2)[2] for line in lines if line.startswith('0::'))
        group = (root / relative.lstrip('/')).resolve()
        if root.resolve() not in (group, *group.parents):
            return None
        candidates = []
        for directory in (group, *group.parents):
            if directory != root and root not in directory.parents:
                break
            try:
                maximum = directory / 'memory.max'
                current = directory / 'memory.current'
                limit, used = int(maximum.read_text().strip()), int(current.read_text().strip())
                if 0 < limit < (1 << 60):
                    candidates.append({'limit_bytes':limit, 'current_bytes':used,
                                       'available_bytes':max(0,limit-used),
                                       'source':str(maximum)+' and '+str(current)})
            except (OSError, ValueError):
                pass
            if directory == root:
                break
        return min(candidates, key=lambda v: v['available_bytes']) if candidates else None
    except (OSError, StopIteration, IndexError):
        return None


def device_info(path: Path | None = None) -> dict[str, Any]:
    system = platform.system().lower()
    termux = 'com.termux' in os.environ.get('PREFIX', '')
    android = termux or bool(os.environ.get('ANDROID_ROOT') or os.environ.get('ANDROID_DATA')) or system == 'android'
    kind = 'android_termux' if termux else ('android_other' if android else {'windows':'windows','darwin':'macos','linux':'linux'}.get(system, 'unknown'))
    machine = platform.machine().lower()
    arch = 'arm64' if machine in ('aarch64','arm64') else ('x86_64' if machine in ('x86_64','amd64') else machine or 'unknown')
    flags: list[str] | None = None
    cpu_model = None
    try:
        content = Path('/proc/cpuinfo').read_text(encoding='utf8', errors='replace').lower()
        cpu_model = next((line.split(':', 1)[1].strip() for line in content.splitlines() if line.startswith('model name') and ':' in line), None)
        flags = [flag for flag, present in [('avx2', 'avx2' in content.split()), ('neon', any(x in content.split() for x in ('neon','asimd')))] if present]
    except OSError:
        pass
    try:
        battery = psutil.sensors_battery()
        battery_present = True if battery is not None else None
    except (AttributeError, OSError, NotImplementedError):
        battery_present = None
    vm = psutil.virtual_memory()
    cg = _cgroup_limit() if system == 'linux' else None
    available = min(vm.available, cg['available_bytes']) if cg else vm.available
    disk_path = (path or Path.cwd()).resolve()
    # Android system free memory is not an application memory limit.
    app_limit = None
    return {'platform': kind, 'device_form_factor': 'phone_likely' if android else ('laptop_likely' if battery_present else 'unknown'),
            'battery_present': battery_present, 'battery_source': 'psutil.sensors_battery; None means unknown',
            'cpu': {'architecture': arch, 'raw_machine': machine, 'model': cpu_model, 'model_source': '/proc/cpuinfo' if cpu_model else 'unavailable', 'logical_cores': os.cpu_count(),
                    'core_source': 'os.cpu_count', 'instruction_flags': flags,
                    'flags_source': '/proc/cpuinfo' if flags is not None else 'unavailable'},
            'memory': {'total_bytes': vm.total, 'available_system_bytes': vm.available,
                       'source': 'psutil.virtual_memory', 'cgroup': cg,
                       'available_for_budget_bytes': available if not android else None,
                       'android_app_limit_bytes': app_limit,
                       'android_app_limit_source': 'unavailable; Android ActivityManager limit not supplied' if android else 'not applicable'},
            'disk': {'path': str(disk_path), 'free_bytes': shutil.disk_usage(disk_path).free, 'source':'shutil.disk_usage'},
            'packages': {name: importlib.util.find_spec(name) is not None for name in ('numpy','psutil','torch','fastapi','safetensors')},
            'packages_source': 'importlib.util.find_spec (presence only)'}


def estimate_bytes(parameters: int, batch: int, context: int, width: int, layers: int) -> int:
    # 16 B/param is a lower bound for FP32 AdamW; activations are a heuristic.
    return 16*parameters + 16*batch*context*width*layers + 4*batch*4*context*context*layers


def _model_spec(target: int, config: ProbeConfig) -> dict[str, Any]:
    # RMSNorm 2-layer decoder, exact parameter formula: 20*d² + (2*vocab+13)*d.
    v = config.vocab_size
    root = (-(2*v+13) + math.sqrt((2*v+13)**2 + 80*target)) / 40
    d = max(16, int(round(root/4))*4)
    return {'architecture':'tiny_transformer', 'vocab_size':v, 'd_model':d, 'n_layers':2,
            'n_heads':4, 'context_length':config.context_length, 'dropout':0.0,
            'norm_type':'rmsnorm','positional_encoding':'sinusoidal'}


def _param_count(spec: dict[str, Any]) -> int:
    d, v = spec['d_model'], spec['vocab_size']
    return 20*d*d + (2*v+13)*d


def _worker(payload: dict, deadline: float, rss_budget: int | None) -> dict:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return {'status':'timeout', 'error':'Overall probe deadline reached'}
    env = os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    # A frozen executable cannot safely be invoked as a Python interpreter.
    if getattr(sys, 'frozen', False):
        return {'status':'unavailable','error':'Subprocess probing is not supported in this frozen build'}
    proc = subprocess.Popen([sys.executable, '-m', 'modelforge.platforms.probe_worker', json.dumps(payload)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    monitored = psutil.Process(proc.pid)
    sampled_peak = 0
    try:
        while proc.poll() is None:
            try:
                rss = monitored.memory_info().rss
                sampled_peak = max(sampled_peak, rss)
                if rss_budget is not None and rss > rss_budget:
                    proc.kill(); proc.communicate()
                    return {'status':'memory_exceeded', 'sampled_peak_rss_bytes':sampled_peak}
            except psutil.Error:
                pass
            if time.monotonic() >= deadline:
                proc.kill(); proc.communicate()
                return {'status':'timeout', 'sampled_peak_rss_bytes':sampled_peak}
            time.sleep(0.01)
        out, err = proc.communicate()
        if proc.returncode != 0:
            return {'status':'failed','error':err[-2000:], 'exit_code':proc.returncode, 'sampled_peak_rss_bytes':sampled_peak}
        result = json.loads(out)
        peak = max(sampled_peak, result.get('peak_rss_bytes_worker') or 0)
        result['peak_rss_bytes'] = peak
        result['rss_source'] = 'max(worker ru_maxrss, parent psutil sampling)' if result.get('peak_rss_bytes_worker') is not None else 'parent psutil sampling (may miss spikes)'
        result['status'] = 'memory_exceeded' if rss_budget is not None and peak > rss_budget else 'ok'
        return result
    finally:
        if proc.poll() is None:
            proc.kill(); proc.communicate()


def classify(candidates: list[dict], config: ProbeConfig) -> dict:
    passed = [c for c in candidates if c.get('status') == 'passed' and c.get('weights_changed') is True]
    best = max(passed, key=lambda c: c['parameters']) if passed else None
    return {'max_params': best['parameters'] if best else None,
            'training_from_scratch': {'tested_candidate': best, 'estimated_seconds_per_epoch': best['step_seconds'] * config.steps_per_epoch if best else None,
                                      'epoch_basis': f'{config.steps_per_epoch} steps; approximation, no dataset supplied'},
            'finetune': 'Not measured; cannot assert a supported size.',
            'inference': 'Not measured; cannot assert a supported size.',
            'remote': 'Use a larger machine for sizes not verified here.',
            'warnings': ['A single step is not full training; actual memory, speed and device temperature may differ.']}


def measure_capabilities(config: ProbeConfig | None = None, path: Path | None = None, low_memory_mode: bool = False) -> dict[str, Any]:
    config = config or ProbeConfig()
    started = time.monotonic()
    deadline = started + config.total_timeout_seconds
    device = device_info(path)
    report: dict[str, Any] = {'schema_version':1, 'device':device, 'config':asdict(config),
                              'measurement_kind':'bounded probe, not full training', 'benchmark':{}, 'candidates':[]}
    available = device['memory']['available_for_budget_bytes']
    if available is None:
        report['status'] = 'unavailable'
        report['reason'] = 'Android application memory limit is unknown; safe probing is disabled.'
        report['classification'] = classify([], config)
        return report
    budget = memory_budget(available, config.reserve_fraction)
    report['memory_budget'] = {'bytes':budget, 'reserve_fraction':config.reserve_fraction,
                               'source':'available system/cgroup memory times configured reserve', 'approximate':True}
    def run(payload):
        return _worker(payload, min(deadline, time.monotonic()+config.per_probe_seconds), budget)
    matmul = run({'kind':'matmul','size':512,'repeats':5})
    report['benchmark']['matmul'] = matmul
    ref_spec = _model_spec(100_000, config)
    reference = run({'kind':'step','spec':ref_spec,'batch_size':config.batch_size}) if matmul['status'] == 'ok' else {'status':'not_run'}
    report['benchmark']['reference_step'] = reference
    if reference.get('status') != 'ok' or not reference.get('weights_changed'):
        report['status']='incomplete'
        report['reason']='Reference measurement failed; candidate selection was not started.'
        report['classification']=classify([],config)
        return report
    for target in config.candidates:
        spec = _model_spec(target, config)
        params = _param_count(spec)
        estimate = estimate_bytes(params, config.batch_size, config.context_length, spec['d_model'], spec['n_layers'])
        row: dict[str, Any] = {'target_parameters':target,'estimated_parameters':params,'spec':spec,
                               'estimated_training_bytes':estimate,'estimate_note':'approximate FP32 AdamW + activation heuristic, excludes interpreter and transient buffers'}
        if time.monotonic() >= deadline:
            row['status']='not_tested_timeout'
        elif estimate >= budget:
            row['status']='rejected_estimate_over_budget'
        else:
            result = run({'kind':'step','spec':spec,'batch_size':config.batch_size})
            row.update(result)
            if result['status'] == 'ok' and result.get('parameters') == params and result.get('weights_changed'):
                row['status']='passed'
            elif result['status'] == 'ok':
                row['status']='failed_validation'
        report['candidates'].append(row)
        if row['status'] in ('timeout','memory_exceeded'):
            # Larger models must not be inferred safe after a resource failure.
            break
    report['classification'] = classify(report['candidates'],config)
    if low_memory_mode:
        report['classification']['warnings'].append('Low Memory Mode is enabled; this probe does not override its safeguards.')
    if device['platform'].startswith('android'):
        report['classification']['warnings'].append('Android resource limits require on-device verification.')
    report['status'] = 'completed' if report['classification']['max_params'] else 'incomplete'
    report['elapsed_seconds'] = time.monotonic()-started
    return report
