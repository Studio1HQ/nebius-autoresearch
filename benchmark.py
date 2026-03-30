"""
benchmark.py — Fixed evaluation harness for Nebius AutoResearch.

Generates 500K synthetic server log entries, runs solve.process(), verifies
correctness against a golden reference, and scores throughput.

DO NOT MODIFY THIS FILE. This is the ground-truth evaluation, equivalent to
prepare.py + evaluate_bpb() in Karpathy's autoresearch.

Usage:
    python benchmark.py

Output:
    ---
    score:              <entries_per_second>   (higher is better, 0 if incorrect)
    processing_time:    <seconds>
    correctness:        pass | FAIL
    num_entries:        500000
"""

import random
import time
import math
import json
import sys
import traceback

# ---------------------------------------------------------------------------
# Fixed constants (DO NOT CHANGE)
# ---------------------------------------------------------------------------

NUM_ENTRIES    = 500_000
SEED           = 42
TIME_BUDGET    = 30   # seconds — kill if exceeded

ENDPOINTS = [
    "/api/users", "/api/orders", "/api/products", "/api/auth/login",
    "/api/search", "/api/feed", "/api/settings", "/api/notifications",
    "/api/payments", "/health",
]
METHODS        = ["GET", "POST", "PUT", "DELETE"]
STATUS_CODES   = [200, 201, 204, 301, 400, 401, 403, 404, 500, 502, 503]
STATUS_WEIGHTS = [ 50,  10,   5,   5,   8,   5,   3,  10,   2,   1,   1]

# ---------------------------------------------------------------------------
# Data generation (deterministic)
# ---------------------------------------------------------------------------

def _generate_ips(n=500):
    rng = random.Random(SEED + 1)
    return [f"{rng.randint(1,254)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"
            for _ in range(n)]

_IPS = _generate_ips()

def generate_logs(n=NUM_ENTRIES):
    """Generate n deterministic log lines. Each line:
       timestamp ip method endpoint status latency size
    """
    rng = random.Random(SEED)
    lines = []
    for _ in range(n):
        h, m, s = rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59)
        ts = f"2024-01-15T{h:02d}:{m:02d}:{s:02d}"
        ip       = rng.choice(_IPS)
        method   = rng.choice(METHODS)
        endpoint = rng.choice(ENDPOINTS)
        status   = rng.choices(STATUS_CODES, weights=STATUS_WEIGHTS)[0]
        latency  = round(rng.expovariate(10), 6)   # mean ~0.1 s
        size     = rng.randint(64, 65536)
        lines.append(f"{ts} {ip} {method} {endpoint} {status} {latency} {size}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Golden reference implementation
# ---------------------------------------------------------------------------

def _p95(values):
    """95th percentile, consistent formula for solve.py to match."""
    s = sorted(values)
    idx = max(0, int(math.ceil(0.95 * len(s))) - 1)
    return round(s[idx], 6)

def compute_reference(log_data):
    from collections import Counter

    lines = log_data.split("\n")
    ep_list, st_list, ip_list, hour_list, size_list = [], [], [], [], []
    latencies_by_ep = {}
    minute_list = []

    for line in lines:
        parts = line.split(" ")
        ts, ip, method, endpoint = parts[0], parts[1], parts[2], parts[3]
        status  = int(parts[4])
        latency = float(parts[5])
        size    = int(parts[6])

        ep_list.append(endpoint)
        st_list.append(status)
        ip_list.append(ip)
        hour_list.append(int(ts[11:13]))
        size_list.append(size)
        minute_list.append(ts[11:16])
        latencies_by_ep.setdefault(endpoint, []).append(latency)

    ip_counter = Counter(ip_list)

    # Top 10 IPs — deterministic tie-breaking: count desc, then IP string asc
    sorted_ips = sorted(ip_counter.items(), key=lambda x: (-x[1], x[0]))
    top_ips = [(ip, c) for ip, c in sorted_ips[:10]]

    # Anomalous IPs
    counts = list(ip_counter.values())
    mean_c = sum(counts) / len(counts)
    stddev_c = (sum((c - mean_c) ** 2 for c in counts) / len(counts)) ** 0.5
    threshold = mean_c + 3 * stddev_c
    anomalous_ips = sorted(ip for ip, c in ip_counter.items() if c > threshold)

    return {
        "endpoint_request_counts": dict(Counter(ep_list)),
        "status_code_counts":     dict(Counter(st_list)),
        "top_ips":                top_ips,
        "avg_latency_by_endpoint": {
            ep: round(sum(lats) / len(lats), 6)
            for ep, lats in latencies_by_ep.items()
        },
        "p95_latency_by_endpoint": {ep: _p95(lats) for ep, lats in latencies_by_ep.items()},
        "peak_minute":    Counter(minute_list).most_common(1)[0][0],
        "error_rate":     round(sum(1 for s in st_list if s >= 400) / len(st_list), 6),
        "total_bytes":    sum(size_list),
        "hourly_counts":  dict(Counter(hour_list)),
        "anomalous_ips":  anomalous_ips,
    }

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _match(ref, res, path, tol=1e-4):
    """Recursively compare reference and result values."""
    if isinstance(ref, tuple):
        ref = list(ref)
    if isinstance(res, tuple):
        res = list(res)

    if isinstance(ref, dict):
        if not isinstance(res, dict):
            print(f"  FAIL [{path}]: expected dict, got {type(res).__name__}")
            return False
        ref_s = {str(k): v for k, v in ref.items()}
        res_s = {str(k): v for k, v in res.items()}
        if ref_s.keys() != res_s.keys():
            missing = ref_s.keys() - res_s.keys()
            extra   = res_s.keys() - ref_s.keys()
            print(f"  FAIL [{path}]: key mismatch  missing={missing}  extra={extra}")
            return False
        return all(_match(ref_s[k], res_s[k], f"{path}.{k}", tol) for k in ref_s)

    if isinstance(ref, list):
        if not isinstance(res, (list, tuple)):
            print(f"  FAIL [{path}]: expected list, got {type(res).__name__}")
            return False
        res = list(res)
        if len(ref) != len(res):
            print(f"  FAIL [{path}]: length {len(ref)} vs {len(res)}")
            return False
        return all(_match(ref[i], res[i], f"{path}[{i}]", tol) for i in range(len(ref)))

    if isinstance(ref, float) or isinstance(res, float):
        try:
            rf, rs = float(ref), float(res)
        except (TypeError, ValueError):
            print(f"  FAIL [{path}]: cannot compare as float")
            return False
        if abs(rf - rs) > tol * max(abs(rf), 1e-10):
            print(f"  FAIL [{path}]: {rf:.6f} vs {rs:.6f}")
            return False
        return True

    if ref != res:
        print(f"  FAIL [{path}]: {ref!r} vs {res!r}")
        return False
    return True


def verify(result, reference):
    expected_keys = set(reference.keys())
    actual_keys   = set(result.keys()) if isinstance(result, dict) else set()
    if expected_keys != actual_keys:
        print(f"  FAIL: key mismatch  expected={expected_keys}  got={actual_keys}")
        return False
    return all(_match(reference[k], result[k], k) for k in reference)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating {NUM_ENTRIES:,} log entries...", flush=True)
    log_data = generate_logs()

    print("Computing reference answers...", flush=True)
    reference = compute_reference(log_data)

    print(f"Running solve.process()...", flush=True)

    try:
        import solve
    except Exception as e:
        print(f"\nFAIL: cannot import solve.py — {type(e).__name__}: {e}")
        traceback.print_exc()
        print("---")
        print("score:              0.0")
        print("processing_time:    0.000")
        print("correctness:        FAIL")
        print(f"num_entries:        {NUM_ENTRIES}")
        sys.exit(1)

    t0 = time.perf_counter()
    try:
        result = solve.process(log_data)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"\nFAIL: solve.process() raised {type(e).__name__}: {e}")
        traceback.print_exc()
        print("---")
        print("score:              0.0")
        print(f"processing_time:    {elapsed:.3f}")
        print("correctness:        FAIL")
        print(f"num_entries:        {NUM_ENTRIES}")
        sys.exit(1)
    elapsed = time.perf_counter() - t0

    if elapsed > TIME_BUDGET:
        print(f"\nFAIL: exceeded time budget ({elapsed:.1f}s > {TIME_BUDGET}s)")
        print("---")
        print("score:              0.0")
        print(f"processing_time:    {elapsed:.3f}")
        print("correctness:        timeout")
        print(f"num_entries:        {NUM_ENTRIES}")
        sys.exit(1)

    correct = verify(result, reference)
    throughput = NUM_ENTRIES / elapsed
    score = throughput if correct else 0.0

    print()
    print("---")
    print(f"score:              {score:.1f}")
    print(f"processing_time:    {elapsed:.3f}")
    print(f"correctness:        {'pass' if correct else 'FAIL'}")
    print(f"entries_per_second: {throughput:.1f}")
    print(f"num_entries:        {NUM_ENTRIES}")
