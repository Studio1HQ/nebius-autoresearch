"""
solve.py — Log processing pipeline.

THIS IS THE ONLY FILE THE AGENT MODIFIES.

Process 500K server log entries and compute analytics. The benchmark
calls process(log_data) and checks every output against a golden reference.

Optimise for speed while keeping outputs bit-exact. Higher entries/second = better score.
"""

import math


def process(log_data: str) -> dict:
    """
    Process raw log data and return analytics results.

    Input: newline-separated log entries, each formatted as:
        timestamp ip method endpoint status latency size
        e.g. "2024-01-15T08:23:45 10.0.1.42 GET /api/users 200 0.045123 342"

    Returns a dict with exactly these keys:
        endpoint_request_counts : dict[str, int]
        status_code_counts      : dict[int, int]
        top_ips                 : list of (ip, count) — top 10 by count desc, ties broken by ip asc
        avg_latency_by_endpoint : dict[str, float]  — rounded to 6 decimals
        p95_latency_by_endpoint : dict[str, float]  — rounded to 6 decimals
        peak_minute             : str like "08:23"   — the minute with the most requests
        error_rate              : float              — fraction of status >= 400, rounded to 6 decimals
        total_bytes             : int
        hourly_counts           : dict[int, int]     — hour (0-23) → count
        anomalous_ips           : list[str]          — sorted IPs with count > mean + 3*stddev
    """
    lines = log_data.split("\n")

    # ── Parse all lines into records ─────────────────────────────────────
    records = []
    for line in lines:
        parts = line.split(" ")
        records.append({
            "timestamp": parts[0],
            "ip":        parts[1],
            "method":    parts[2],
            "endpoint":  parts[3],
            "status":    int(parts[4]),
            "latency":   float(parts[5]),
            "size":      int(parts[6]),
        })

    # ── 1. Endpoint request counts ───────────────────────────────────────
    endpoint_request_counts = {}
    for r in records:
        ep = r["endpoint"]
        if ep in endpoint_request_counts:
            endpoint_request_counts[ep] += 1
        else:
            endpoint_request_counts[ep] = 1

    # ── 2. Status code counts ────────────────────────────────────────────
    status_code_counts = {}
    for r in records:
        s = r["status"]
        if s in status_code_counts:
            status_code_counts[s] += 1
        else:
            status_code_counts[s] = 1

    # ── 3. Top 10 IPs ───────────────────────────────────────────────────
    ip_counts = {}
    for r in records:
        ip = r["ip"]
        if ip in ip_counts:
            ip_counts[ip] += 1
        else:
            ip_counts[ip] = 1
    sorted_ips = sorted(ip_counts.items(), key=lambda x: (-x[1], x[0]))
    top_ips = [(ip, count) for ip, count in sorted_ips[:10]]

    # ── 4. Average latency by endpoint ───────────────────────────────────
    latency_sums = {}
    latency_counts = {}
    for r in records:
        ep  = r["endpoint"]
        lat = r["latency"]
        if ep in latency_sums:
            latency_sums[ep] += lat
            latency_counts[ep] += 1
        else:
            latency_sums[ep] = lat
            latency_counts[ep] = 1
    avg_latency_by_endpoint = {}
    for ep in latency_sums:
        avg_latency_by_endpoint[ep] = round(latency_sums[ep] / latency_counts[ep], 6)

    # ── 5. P95 latency by endpoint ───────────────────────────────────────
    latencies_by_ep = {}
    for r in records:
        ep = r["endpoint"]
        if ep not in latencies_by_ep:
            latencies_by_ep[ep] = []
        latencies_by_ep[ep].append(r["latency"])
    p95_latency_by_endpoint = {}
    for ep, lats in latencies_by_ep.items():
        sorted_lats = sorted(lats)
        idx = max(0, int(math.ceil(0.95 * len(sorted_lats))) - 1)
        p95_latency_by_endpoint[ep] = round(sorted_lats[idx], 6)

    # ── 6. Peak minute ───────────────────────────────────────────────────
    minute_counts = {}
    for r in records:
        minute = r["timestamp"][11:16]
        if minute in minute_counts:
            minute_counts[minute] += 1
        else:
            minute_counts[minute] = 1
    peak_minute = max(minute_counts.items(), key=lambda x: x[1])[0]

    # ── 7. Error rate ────────────────────────────────────────────────────
    errors = 0
    for r in records:
        if r["status"] >= 400:
            errors += 1
    error_rate = round(errors / len(records), 6)

    # ── 8. Total bytes ───────────────────────────────────────────────────
    total_bytes = 0
    for r in records:
        total_bytes += r["size"]

    # ── 9. Hourly counts ─────────────────────────────────────────────────
    hourly_counts = {}
    for r in records:
        hour = int(r["timestamp"][11:13])
        if hour in hourly_counts:
            hourly_counts[hour] += 1
        else:
            hourly_counts[hour] = 1

    # ── 10. Anomalous IPs ────────────────────────────────────────────────
    counts = list(ip_counts.values())
    mean_count = sum(counts) / len(counts)
    variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
    stddev = variance ** 0.5
    threshold = mean_count + 3 * stddev
    anomalous_ips = sorted(ip for ip, c in ip_counts.items() if c > threshold)

    return {
        "endpoint_request_counts": endpoint_request_counts,
        "status_code_counts":      status_code_counts,
        "top_ips":                 top_ips,
        "avg_latency_by_endpoint": avg_latency_by_endpoint,
        "p95_latency_by_endpoint": p95_latency_by_endpoint,
        "peak_minute":             peak_minute,
        "error_rate":              error_rate,
        "total_bytes":             total_bytes,
        "hourly_counts":           hourly_counts,
        "anomalous_ips":           anomalous_ips,
    }
