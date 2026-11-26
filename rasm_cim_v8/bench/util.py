"""v0.8 bench shared helpers: subprocess isolation, RSS, latency stats."""
from __future__ import annotations
import json, os, statistics, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rss_mib() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cur_rss_mib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def stats_ms(times_s: list[float]) -> dict:
    ts = sorted(t * 1000.0 for t in times_s)
    n = len(ts)
    return {"n": n,
            "p50_ms": statistics.median(ts),
            "p95_ms": ts[min(n - 1, int(0.95 * n))],
            "max_ms": ts[-1],
            "mean_ms": sum(ts) / n}


def in_subprocess(code: str, timeout: int = 900) -> dict:
    """Run measurement code in a fresh interpreter; code must print one
    JSON line. Returns parsed dict or {"error": ...}."""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=ROOT, timeout=timeout)
    if p.returncode != 0:
        return {"error": p.stderr[-500:]}
    last = p.stdout.strip().splitlines()[-1]
    try:
        return json.loads(last)
    except Exception:
        return {"error": f"bad json: {last[:200]}"}


def cpu_pair():
    t = os.times()
    return t.user + t.system


def save(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print("saved", path)
