# AutoResearch Beyond ML: Letting an AI Agent Optimize Real-World Data Pipelines

I built an AI agent that rewrites Python code, executes it against real data, measures performance, and keeps only the improvements. Then I let it run.

In under 10 minutes, it made a data analytics pipeline 7x faster — processing half a million real NYC taxi trips — without me touching a single line.

No synthetic benchmarks. No toy problems. Real data, real code, real speedups.

![Dashboard showing score progression from baseline to 7x improvement](assets/dashboard-overview.png)
*The live dashboard tracking the agent's optimization progress in real time.*

---

## Why This Is Different

Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) introduced a powerful pattern: give an AI a script, a real metric, and a loop — then walk away. The agent edits code, runs it, checks the score, keeps what works, reverts what doesn't.

Since then, most implementations have stayed in ML territory — optimising training loops on GPUs. Varun Mathur scaled it to 237 parallel agents, the SkyPilot team ran 910 experiments across 16 GPUs. Impressive engineering, but they all answer the same question: can AI make ML training faster?

We're asking a different one: **can this pattern make everyday data engineering faster?**

Not ML training. Not synthetic benchmarks. A real data pipeline processing real NYC taxi trip records — the kind of code data engineers write every day. The kind that processes CSVs, computes business metrics, and runs on a laptop.

The answer turns out to be yes, and the optimisation path is surprisingly similar to how an experienced developer would approach it — just faster.

---

## What We're Building

An autonomous agent that optimises a Python analytics pipeline operating on **500,000 real NYC Yellow Taxi trip records** from the [NYC TLC open data portal](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

The pipeline computes 9 real business metrics: revenue by payment type, average tip by hour, trip distance distributions, busiest hours, top routes, fare-per-mile trends, and daily revenue breakdowns.

The agent's job: make it faster without breaking any of the numbers.

---

## The Pattern

Karpathy's structure has three files. We keep the same separation:

| File | Role | Who touches it |
| --- | --- | --- |
| `prepare_data.py` | Downloads real data from NYC TLC | Nobody (run once) |
| `benchmark.py` | Loads data, runs code, verifies, scores | Nobody |
| `solve.py` | The analytics pipeline | The agent |
| `program.md` | Standing instructions | The human, once |

The loop: read code → propose one change → run against real data → score → keep or revert → repeat.

What makes it trustworthy: the score comes from real execution against real data. The agent can't talk its way to a better number. The code has to actually get faster while producing identical results.

|  | Karpathy | This project |
| --- | --- | --- |
| Domain | ML training | Real-world data analytics |
| Data | Generated during training | 500K real NYC taxi records |
| Agent edits | `train.py` | `solve.py` |
| Evaluation | `prepare.py` | `benchmark.py` |
| Metric | val_bpb (lower is better) | trips/sec (higher is better) |
| Hardware | GPU required | Any laptop |
| Time per round | ~5 minutes | ~30 seconds |

---

## Project Structure

```
nebius-autoresearch/
├── nebius_agent.py    ← the autonomous optimization loop
├── benchmark.py       ← fixed evaluation harness, never modified
├── solve.py           ← the only file the agent touches
├── prepare_data.py    ← downloads real NYC taxi data (run once)
├── dashboard.py       ← live web dashboard (Flask)
├── program.md         ← standing instructions for the agent
├── requirements.txt   ← Python dependencies
├── .env.example       ← environment variable template
├── data/
│   └── taxi_trips.csv ← 500K real trip records (~34 MB, generated)
├── templates/
│   └── index.html     ← dashboard UI (Chart.js)
├── assets/            ← screenshots for the blog
└── results.tsv        ← experiment log (auto-generated)
```

---

## Prerequisites

```bash
git clone https://github.com/VarshithKrishna14/nebius-autoresearch.git
cd nebius-autoresearch
pip install -r requirements.txt
export NEBIUS_API_KEY="your-key-here"   # Windows: $env:NEBIUS_API_KEY = "..."
```

Then prepare the data (one-time download, ~40 seconds):

```bash
python prepare_data.py
```

This downloads January 2024 Yellow Taxi trip records from the NYC TLC open data portal, samples 500K clean rows, and saves them as a CSV. The data is real — actual pickup times, real fares, real tip amounts, real locations.

---

## Step 1 — The Benchmark

`benchmark.py` loads the real taxi CSV, calls `solve.process()`, and validates 9 outputs against a golden reference computed from the same data.

The analytics cover real business questions:

1. **Revenue by payment type** — cash vs credit vs other
2. **Average tip by hour** — when do people tip more?
3. **Passenger distribution** — solo riders vs groups
4. **Trip distance stats** — mean, P50, P95
5. **Duration P95** — how long do the longest trips take?
6. **Busiest hours** — top 5 by trip count
7. **Top routes** — most common pickup-dropoff pairs
8. **Fare per mile by hour** — pricing trends across the day
9. **Daily revenue** — revenue over the month

If any output is wrong, score is 0. If all pass:

```
score = 500,000 / processing_time_seconds
```

Running the baseline:

```
score:              24426.7
processing_time:    20.469
correctness:        pass
trips_per_second:   24426.7
num_trips:          500000
```

![Terminal showing baseline benchmark output](assets/baseline-benchmark.png)
*The baseline score — the naive implementation processes ~24K trips per second. Results vary by hardware.*

Twenty seconds to process 500K rows. There's a lot of room to improve.

---

## Step 2 — The Naive Baseline

`solve.py` starts as correct-but-slow code — the kind you'd write to get the numbers right before caring about performance:

- **500K dict objects** — one per CSV row
- **9 separate passes** over all records
- **`datetime.strptime()` called 500K+ times** — the biggest bottleneck
- Full array sort for every percentile
- Manual dict-based counting instead of `Counter`

Baseline: **~24K trips/sec** on a standard laptop. The `strptime` calls alone account for roughly half the runtime.

The optimisation path from here is real and multi-layered:

| Round | Optimisation | Expected impact |
| --- | --- | --- |
| Early | `Counter` + combined loops | 1.3-1.5x |
| Middle | Replace `strptime` with string slicing | 2-3x |
| Late | Tuples instead of dicts, single-pass pipeline | 2-3x on top |
| Advanced | Numpy for numeric aggregations | additional gains |

---

## Step 3 — The Reasoning Model

We're using **Qwen3-235B** through [Nebius AI](https://nebius.com), specifically the `Qwen3-235B-A22B-Thinking-2507` variant. It's a reasoning model — it works through *why* code might be slow before proposing a fix, which produces better proposals than models that just pattern-match.

Like other thinking models, it emits `<think>...</think>` reasoning blocks that must be stripped from every response:

```python
def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()
```

Set `max_tokens` to at least **8192**. The model spends 2000-3000 tokens thinking before writing code. Too low and you get truncated functions with syntax errors.

---

## Step 4 — Proposing a Change

Each round, the agent gets the current `solve.py`, the `program.md` instructions, and a summary of recent experiments:

```python
def propose_change(history):
    prompt = f"""
You are an expert Python performance engineer.

{open("program.md").read()}

Best score so far: {best_score(history):.1f} trips/second
Recent experiments: {format_history(history, last_n=10)}
Current solve.py: {open("solve.py").read()}

Propose ONE focused optimisation. Return the complete updated solve.py.

DESCRIPTION: <one sentence>
CODE:
<complete solve.py>
    """

    result = client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B-Thinking-2507",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
    )

    raw = strip_thinking(result.choices[0].message.content)
    return extract_code(raw), parse_description(raw)
```

One change per round keeps progress traceable. The description forces the model to articulate what it changed.

---

## Step 5 — Running the Benchmark

```python
def run_benchmark() -> float:
    proc = subprocess.run(
        ["python", "benchmark.py"],
        capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        return 0.0
    return parse_score(proc.stdout)
```

No AI evaluation. The benchmark runs against real taxi data, either passes all 9 correctness checks and returns a number, or it doesn't.

---

## Step 6 — The Loop

```python
def run_agent(n_experiments=None):
    best_score = run_benchmark()  # baseline, no API call
    history = []

    while True:
        new_code, description = propose_change(history)

        write("solve.py", new_code)
        commit = git_commit(f"experiment: {description}")

        score = run_benchmark()

        if score > best_score:
            best_score = score
            status = "keep"
        else:
            git_reset_hard()
            status = "revert" if score > 0 else "crash"

        history.append({"score": score, "description": description, "status": status})
```

Every kept improvement is a real git commit on a dedicated branch.

---

## Running It

```bash
# Install dependencies
pip install -r requirements.txt

# One-time setup
python prepare_data.py

# Check the baseline
python benchmark.py

# Run 20 experiments (real-time API calls)
python nebius_agent.py --n-experiments 20

# Run with batch inference (50% cheaper)
python nebius_agent.py --n-experiments 50 --batch

# Run overnight
python nebius_agent.py
```

You can also launch the live dashboard to monitor progress in real time:

```bash
python dashboard.py
# Open http://localhost:5000
```

![Live dashboard with agent running](assets/dashboard-live.png)
*The dashboard auto-refreshes every 3 seconds, showing score chart, experiment history, current code, and live logs.*

---

## What Actually Happened

Here's what a session with tuned parameters and enough rounds produces — 18 experiments, about 9 minutes:

| Round | Score | Status | What changed |
| --- | --- | --- | --- |
| Baseline | 24,427 | keep | naive implementation |
| 1 | 32,000 | keep | `Counter` + combined counting loops |
| 2 | 28,000 | revert | early exit attempt regressed |
| 3 | 65,000 | keep | replaced `datetime.strptime` with string slicing |
| 4 | 95,000 | keep | tuples instead of dicts for parsed rows |
| 5 | 140,000 | keep | single-pass pipeline |
| ... | ... | ... | some regressions, then recovery |
| Best | 175,000 | keep | ~7x over baseline |

![Score progression chart showing the optimization trajectory](assets/score-progression.png)
*Score progression across experiments. Green dots = kept improvements. Yellow = discarded (no improvement). Red = crashed.*

![Terminal output showing agent running experiments](assets/agent-running.png)
*The agent running autonomously — proposing changes, committing, benchmarking, and deciding keep/revert.*

The biggest single jump — round 3, replacing `datetime.strptime()` with string arithmetic — is the kind of thing a senior developer would spot immediately. The model finds it because it reasons about *where the time is going* before proposing a fix.

The structural changes — tuples over dicts, single-pass aggregation — compound across rounds. Nobody told the agent the order. It discovered it.

### Results Summary

After a full run of 18 experiments:

- **Baseline**: ~24K trips/second (naive implementation)
- **Best**: ~175K trips/second
- **Speedup**: ~7.2x
- **Kept changes**: 6 out of 18 experiments
- **Crashes**: 3 (syntax errors or incorrect outputs)
- **Total time**: ~9 minutes (including API calls + benchmarking)
- **Total cost**: ~$0.36 in API calls (real-time mode)

![Dashboard stats panel showing final results](assets/final-results.png)
*Final results on the dashboard — 7.2x speedup achieved autonomously.*

---

## Why the Loop Is Trustworthy

The model isn't deeply understanding your data pipeline. It recognises patterns from Python performance writing it's seen before — multiple passes are suspicious, `strptime` is slow, sorting a full array for one percentile is wasteful.

What makes the loop trustworthy isn't the model — it's the benchmark. Every proposed change runs against 500,000 real taxi records and must produce identical analytics. A bad optimisation that gives different numbers gets a score of zero and gets reverted.

That's the core of autoresearch. Without the feedback loop, you're relying on the model to judge its own output. With it, reality does the judging — using real data.

---

## Why Nebius

**The model.** We're using Qwen3-235B-A22B-Thinking via the Nebius Token Factory API. It's a reasoning model with a 262K context window. The OpenAI-compatible API means one import and one base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507"
```

No new SDK. If you've used the OpenAI client before, nothing changes.

**The cost.** Nebius batch inference cuts cost in half. Instead of individual real-time API calls, submit all proposals as a single async JSONL job:

```
Upload JSONL → Submit batch job → Poll until done → Download results
```

50% cheaper, no rate limit consumption. The `--batch` flag in our agent handles this automatically:

```bash
python nebius_agent.py --n-experiments 50 --batch
```

Under the hood, batch mode builds a JSONL file with all N proposals, uploads it via `client.files.create()`, submits a batch job via `client.batches.create()`, polls until complete, then downloads and evaluates each proposal sequentially. Each proposal gets a different diversity hint so the model doesn't propose the same optimisation N times.

| Run size | Real-time cost | Batch cost | Saving |
| --- | --- | --- | --- |
| 20 rounds | ~$0.40 | ~$0.20 | $0.20 |
| 50 rounds | ~$1.00 | ~$0.50 | $0.50 |
| 200 rounds | ~$4.00 | ~$2.00 | $2.00 |

---

## `program.md` — What to Put Here

```markdown
# Task
Optimize solve.py to maximize trips/sec on benchmark.py.

# Rules
- Only modify solve.py
- Keep the process(csv_data) signature unchanged
- All 9 correctness checks must pass
- One focused change per round

# Known bottlenecks
- datetime.strptime() called 500K+ times
- 500K dict objects created during parsing
- 9 separate passes over all records

# Things worth trying
- Replace strptime with string slicing for timestamp math
- Counter instead of manual dict counting
- Tuples instead of dicts for parsed rows
- Single-pass aggregation pipeline
- heapq.nlargest instead of full sort for top-N

# Do not
- Hardcode outputs
- Modify benchmark.py
- Repeat failed experiments
```

---

## The Dashboard

The project includes a live web dashboard built with Flask and Chart.js. It's not just a monitoring tool — you can start and stop the agent directly from the browser.

![Dashboard full view with all panels](assets/dashboard-full.png)
*The complete dashboard: stats, score chart, agent controls, experiment table, code viewer, and live log.*

Key features:
- **Score progression chart** — every experiment plotted with color-coded status (green = kept, yellow = discarded, red = crashed)
- **Stats panel** — baseline score, current best, speedup multiplier, experiment counts
- **Agent controls** — enter your API key, set experiment count, toggle batch mode, start/stop
- **Code viewer** — syntax-highlighted view of the current best `solve.py`
- **Live log** — last 60 lines of benchmark output, auto-refreshing every 3 seconds

```bash
python dashboard.py
# Open http://localhost:5000
```

---

## Where to Take This

The loop doesn't care what it's optimising. Swap the data and the benchmark:

**SQL queries** — benchmark measures execution time against a real database, agent rewrites the query.

**API handlers** — benchmark runs a load test, agent rewrites the handler. Keep a golden response set for correctness.

**ETL pipelines** — same structure, different input format and output checks.

**Any CSV processing** — download a different dataset, define different metrics, let the agent run. The taxi data is just one example. Try it on your own data.

---

## Three Things I'd Tell You Before Starting

**Set `max_tokens` to 8192 minimum.** Reasoning models think before they code. With 4096, you get truncated functions and syntax errors that look like model failures but are actually a config issue.

**Use real data.** Synthetic benchmarks are easy to build but hard to trust. When the agent optimises code running against real taxi records, the improvements transfer to production. When it optimises code running against `random.Random(42)` output, you're optimising for a specific random distribution that doesn't exist anywhere.

**Run it long enough.** Five rounds won't tell you much. Twenty starts to show a pattern. The structural improvements — the ones that actually move the score — tend to show up after the easy wins are committed.
