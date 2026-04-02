# Screenshot Assets

Place screenshots here before publishing the blog. The `BLOG_TUTORIAL.md` references these files.

## Required Screenshots

| Filename | What to capture |
|----------|----------------|
| `dashboard-overview.png` | Full dashboard after a completed run — shows score chart, stats, table with multiple experiments |
| `baseline-benchmark.png` | Terminal output of `python benchmark.py` on the naive baseline (shows ~24K trips/sec) |
| `score-progression.png` | The Chart.js score progression graph from the dashboard mid-run, showing the improvement curve |
| `agent-running.png` | Terminal output of `python nebius_agent.py` mid-run — showing EXPERIMENT N, the proposals, IMPROVED lines |
| `final-results.png` | Dashboard stats panel zoomed in — Baseline, Best Score, Speedup, Experiments |
| `dashboard-live.png` | Full dashboard while agent is actively running — status dot pulsing green, log panel showing recent output |
| `dashboard-full.png` | Full dashboard after completion with all panels visible |

## How to Take Them

1. Run `python prepare_data.py` (one time)
2. Run `python dashboard.py` and open http://localhost:5000
3. Start the agent from the dashboard with 20 experiments
4. Capture screenshots at the right moments

For `agent-running.png`, run directly in terminal:
```bash
python nebius_agent.py --n-experiments 5
```
and screenshot after the first few experiments complete.

## Recommended Tool

Use your OS snipping tool or any screen capture tool. Recommended size: 1280×800 or wider.
On Windows: Win + Shift + S (Snipping Tool)
On macOS: Cmd + Shift + 4
