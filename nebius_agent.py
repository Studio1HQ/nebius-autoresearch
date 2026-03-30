"""
nebius_agent.py — Nebius-powered AutoResearch agent.

Autonomously optimises solve.py by calling DeepSeek-R1 on Nebius to propose
code changes, running the benchmark for real scores, and keeping only the
improvements. Same loop pattern as Karpathy's autoresearch, different domain.

Usage:
    export NEBIUS_API_KEY="your-key"

    python nebius_agent.py --setup-branch mar30 --n-experiments 20
    python nebius_agent.py                      # run indefinitely
    python nebius_agent.py --dry-run             # see proposals without running
"""

from openai import OpenAI
import subprocess, os, re, time, argparse

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

MODEL       = "deepseek-ai/DeepSeek-R1-0528"
SOLVE_PY    = "solve.py"
RESULTS_TSV = "results.tsv"
RUN_LOG     = "run.log"
RUN_TIMEOUT = 90  # seconds: 30s budget + data gen + overhead

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*",          "", text, flags=re.DOTALL)
    return text.strip()

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def parse_metric(log: str, key: str) -> float | None:
    m = re.search(rf"^{re.escape(key)}:\s+(\S+)", log, re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Git helpers
# ─────────────────────────────────────────────────────────────────────────────

def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()

def git_commit(message: str) -> str:
    git("add", SOLVE_PY)
    git("commit", "-m", message)
    return git("rev-parse", "--short", "HEAD")

def git_reset_hard() -> None:
    git("reset", "--hard", "HEAD~1")

def setup_branch(tag: str) -> None:
    branch = f"autoresearch/{tag}"
    git("checkout", "-b", branch)
    print(f"✅ Created branch: {branch}")

# ─────────────────────────────────────────────────────────────────────────────
# Results log
# ─────────────────────────────────────────────────────────────────────────────

HEADER = "commit\tscore\tprocessing_time\tstatus\tdescription\n"

def ensure_results_tsv() -> None:
    if not os.path.exists(RESULTS_TSV):
        write_file(RESULTS_TSV, HEADER)

def append_result(commit: str, score: float, proc_time: float,
                  status: str, description: str) -> None:
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(f"{commit}\t{score:.1f}\t{proc_time:.3f}\t{status}\t{description}\n")

def read_results() -> list[dict]:
    if not os.path.exists(RESULTS_TSV):
        return []
    rows = []
    for line in read_file(RESULTS_TSV).strip().splitlines()[1:]:
        parts = line.split("\t", 4)
        if len(parts) == 5:
            rows.append({
                "commit": parts[0], "score": float(parts[1]),
                "processing_time": float(parts[2]),
                "status": parts[3], "description": parts[4],
            })
    return rows

def best_score(history: list[dict]) -> float:
    valid = [r["score"] for r in history if r["score"] > 0]
    return max(valid) if valid else 0.0

def format_history(history: list[dict]) -> str:
    if not history:
        return "  (none yet — this will be the baseline run)"
    lines = []
    for r in history[-20:]:
        lines.append(
            f"  {r['commit']}  score={r['score']:.1f}  "
            f"time={r['processing_time']:.3f}s  [{r['status']}]  {r['description']}"
        )
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Nebius agent: propose a change
# ─────────────────────────────────────────────────────────────────────────────

def propose_change(history: list[dict]) -> tuple[str, str]:
    current_solve = read_file(SOLVE_PY)
    program       = read_file("program.md") if os.path.exists("program.md") else ""

    best = best_score(history)
    is_baseline = best == 0.0

    status_str = ("No experiments yet — this is the baseline run. "
                  "Return solve.py EXACTLY as-is, unchanged." if is_baseline
                  else f"Best score so far: {best:.1f} entries/second")

    prompt = f"""You are an expert Python performance engineer. You are running autonomous
experiments to optimise a log-processing pipeline for maximum throughput.

## Instructions
{program}

## Current status
{status_str}

## Experiment history
{format_history(history)}

## Current solve.py (the ONLY file you modify)
```python
{current_solve}
```

## Your task
{"Return solve.py EXACTLY as-is for the baseline measurement." if is_baseline else "Propose ONE focused optimisation. Do NOT repeat failed experiments."}

Rules:
- Keep the function signature: process(log_data: str) -> dict
- All 10 output keys must remain correct — benchmark verifies every value
- Only use Python standard library (plus numpy if you want)
- Make ONE change with a clear performance rationale

Output EXACTLY this format:
DESCRIPTION: <one sentence describing the change>
CODE:
<complete modified solve.py — raw Python, no markdown fences>"""

    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    raw = strip_thinking(result.choices[0].message.content)

    desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n|$)", raw)
    description = desc_match.group(1).strip() if desc_match else "experiment"

    code_match = re.search(r"CODE:\s*\n(.*)", raw, re.DOTALL)
    if not code_match:
        raise ValueError("Could not parse CODE section from model response")
    code = code_match.group(1).strip()
    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$",             "", code)

    return code, description

# ─────────────────────────────────────────────────────────────────────────────
# Run the benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark() -> tuple[float | None, float | None]:
    """Run benchmark.py, return (score, processing_time) or (None, None) on failure."""
    print("  ⚙️  Running benchmark...", flush=True)
    t0 = time.time()

    try:
        with open(RUN_LOG, "w") as log_f:
            proc = subprocess.Popen(
                ["python", "benchmark.py"],
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            proc.wait(timeout=RUN_TIMEOUT)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            log_tail = "\n".join(read_file(RUN_LOG).strip().splitlines()[-15:])
            print(f"  ❌ Exit code {proc.returncode} after {elapsed:.0f}s")
            print(f"  Last lines:\n{log_tail}")
            return None, None

        print(f"  ✅ Finished in {elapsed:.0f}s")
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  ❌ Timed out after {RUN_TIMEOUT}s")
        return None, None

    log = read_file(RUN_LOG)
    return parse_metric(log, "score"), parse_metric(log, "processing_time")

# ─────────────────────────────────────────────────────────────────────────────
# Main agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(n_experiments: int | None = None, dry_run: bool = False) -> None:
    print("=" * 60)
    print("NEBIUS AUTORESEARCH AGENT — Log Processing Pipeline")
    print(f"Model:       {MODEL}")
    print(f"Experiments: {'∞ (Ctrl-C to stop)' if n_experiments is None else n_experiments}")
    print(f"Dry run:     {dry_run}")
    print("=" * 60)

    ensure_results_tsv()
    history = read_results()
    exp_num = 0

    while n_experiments is None or exp_num < n_experiments:
        exp_num += 1
        print(f"\n{'─' * 60}")
        print(f"EXPERIMENT {exp_num}" + (f"/{n_experiments}" if n_experiments else ""))
        print(f"{'─' * 60}")

        original_solve = read_file(SOLVE_PY)

        # 1. Ask Nebius to propose a change
        print("🤖 Asking Nebius for a change...", flush=True)
        try:
            new_solve, description = propose_change(history)
        except Exception as e:
            print(f"❌ Nebius API error: {e}")
            continue

        print(f"📝 Proposed: {description}")

        orig_lines = set(original_solve.splitlines())
        new_lines  = set(new_solve.splitlines())
        print(f"   Diff: +{len(new_lines - orig_lines)} / -{len(orig_lines - new_lines)} lines")

        if dry_run:
            print("   [DRY RUN] Skipping execution.")
            continue

        # 2. Write + commit
        write_file(SOLVE_PY, new_solve)
        try:
            commit_hash = git_commit(f"experiment: {description}")
        except subprocess.CalledProcessError:
            print("❌ Git commit failed (no changes?)")
            write_file(SOLVE_PY, original_solve)
            continue
        print(f"📦 Committed: {commit_hash}")

        # 3. Run the benchmark
        score, proc_time = run_benchmark()
        proc_time = proc_time or 0.0
        current_best = best_score(history)

        # 4. Keep or revert
        if score is None or score == 0.0:
            status = "crash"
            print("💥 CRASH or FAIL — reverting")
            git_reset_hard()
            append_result(commit_hash, 0.0, proc_time, "crash", description)

        elif score > current_best:
            status = "keep"
            delta = score - current_best
            pct   = 100 * delta / current_best if current_best > 0 else 0
            print(f"✅ IMPROVED  {current_best:.1f} → {score:.1f}  (+{pct:.1f}%)")
            append_result(commit_hash, score, proc_time, "keep", description)

        else:
            status = "discard"
            print(f"❌ NO IMPROVEMENT  got={score:.1f}  best={current_best:.1f} — reverting")
            git_reset_hard()
            append_result(commit_hash, score, proc_time, "discard", description)

        history = read_results()

        # Summary
        kept    = sum(1 for r in history if r["status"] == "keep")
        crashed = sum(1 for r in history if r["status"] == "crash")
        b = best_score(history)
        if b > 0:
            print(f"\n📊 {exp_num} experiments | kept={kept} discard={exp_num-kept-crashed} "
                  f"crash={crashed} | best score={b:.1f}")

    # Final summary
    print("\n" + "=" * 60)
    print("AUTORESEARCH COMPLETE")
    history = read_results()
    baseline = next((r for r in history if "baseline" in r["description"].lower()), None)
    b = best_score(history)
    if baseline and baseline["score"] > 0 and b > 0:
        speedup = b / baseline["score"]
        print(f"Baseline : {baseline['score']:.1f} entries/sec  ({baseline['processing_time']:.3f}s)")
        print(f"Best     : {b:.1f} entries/sec")
        print(f"Speedup  : {speedup:.2f}x")
    elif b > 0:
        print(f"Best score: {b:.1f} entries/sec")
    print(f"Full log : {RESULTS_TSV}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nebius AutoResearch Agent")
    parser.add_argument("--n-experiments", type=int, default=None,
                        help="Number of experiments (default: run forever)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Propose changes without running the benchmark")
    parser.add_argument("--setup-branch", type=str, default=None, metavar="TAG",
                        help="Create autoresearch/<TAG> branch before starting")
    args = parser.parse_args()

    if args.setup_branch:
        setup_branch(args.setup_branch)

    run_agent(n_experiments=args.n_experiments, dry_run=args.dry_run)
