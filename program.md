# Nebius AutoResearch — Log Processing Pipeline

This is an autonomous research loop: an LLM agent iteratively optimises a
data-processing pipeline for maximum throughput, scored by a fixed benchmark.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch),
but applied to CPU-bound data processing instead of GPU-based ML training.

## Project structure

```
benchmark.py    — fixed evaluation harness (DO NOT MODIFY)
solve.py        — the file you modify (log processing pipeline)
program.md      — these instructions (read by the agent)
results.tsv     — experiment log (append-only, not committed)
```

## Setup

1. **Create a branch**: `git checkout -b autoresearch/<tag>` from main.
2. **Read the files**: `benchmark.py`, `solve.py`, and this file.
3. **Run the baseline**: `python benchmark.py` — record the score.
4. **Begin the loop.**

## The task

`benchmark.py` generates **500,000 synthetic server log entries** and calls
`solve.process(log_data)`. Your job: make `process()` return the correct
results as fast as possible.

**The goal: maximise the `score` (entries per second). Higher is better.**

## What you CAN do

- Modify `solve.py` — this is the ONLY file you edit.
- Use any Python standard library (`collections`, `re`, `itertools`, `numpy` if installed, etc.).
- Restructure the code however you want: change data structures, algorithms,
  number of passes, parallelism, caching — everything inside `solve.py` is fair game.

## What you CANNOT do

- Modify `benchmark.py`. It is read-only.
- Change the function signature: `process(log_data: str) -> dict`.
- Return incorrect results. If ANY output is wrong, score = 0.
- Exceed the 30-second time budget.
- Install new packages beyond the standard library (and numpy/openai if present).

## The metric

```
---
score:              <entries_per_second>   ← THIS IS WHAT YOU OPTIMISE
processing_time:    <seconds>
correctness:        pass | FAIL
entries_per_second: <throughput>
num_entries:        500000
```

Higher `score` = better. If correctness fails, score is 0 regardless of speed.

## Simplicity criterion

All else being equal, simpler is better. A 5% speed improvement that adds
30 lines of convoluted code? Marginal. A 5% improvement from deleting code?
Definitely keep. An improvement of ~0% but much cleaner code? Keep.

## Logging results

Append each experiment to `results.tsv` (tab-separated):

```
commit	score	processing_time	status	description
a1b2c3d	125000.0	4.000	keep	baseline
b2c3d4e	180000.0	2.778	keep	single-pass with Counter
c3d4e5f	0.0	0.000	crash	numpy import error
```

## The experiment loop

LOOP FOREVER:

1. Read `solve.py` and the experiment history.
2. Propose ONE focused optimisation with a clear rationale.
3. Edit `solve.py`, git commit.
4. Run `python benchmark.py > run.log 2>&1`
5. Parse the score: `grep "^score:" run.log`
6. If score improved → keep the commit.
7. If score is equal/worse → `git reset --hard HEAD~1`
8. Log the result to `results.tsv`.
9. Repeat.

**NEVER STOP.** The human may be away. Keep experimenting until interrupted.

## Optimisation ideas (non-exhaustive)

- Avoid creating a list of dicts — parse directly into arrays or process in-line.
- Combine multiple counting passes into a single pass.
- Use `collections.Counter` for counting.
- Use `heapq.nlargest` for top-K instead of full sort.
- Use tuples or parallel arrays instead of dicts for records.
- Avoid re-parsing: extract fields once, reuse everywhere.
- Pre-split all lines with a list comprehension.
- Use numpy for statistical computations (percentile, stddev).
- Explore `str.split` vs manual index-based parsing.
- Consider `multiprocessing` for independent computations.
