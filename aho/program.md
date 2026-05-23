# AHO Researcher Program

> Modeled after `karpathy/autoresearch/program.md`.
> This file is the instruction set for the AHO researcher agent.

---

## Setup

Before starting the experiment loop, verify the following:

1. **Read** `aho/harness_config.json` for the current strategy and version number.
2. **Check the endpoint** — run:
   ```
   curl http://localhost:1234/v1/models
   ```
   If it fails, check that LM Studio or Ollama is running before proceeding.
3. **Read the last 10 lines** of `aho/results.jsonl` to understand what's been tried.
   If the file is empty or missing, this is a fresh start — version 0 / `baseline` run first.
4. **Verify** that `aho/bug_tracker.jsonl` exists (created on first run).
5. **Confirm** and begin the loop.

---

## Mutation Rules

You are the edit agent for `aho/harness_config.json["strategy"]`.

- **Only modify** fields inside the `"strategy"` object — nothing else.
- **One change per iteration** — do not bundle multiple ideas.
- **Targeted changes only** — base every mutation on the most frequent `failure_modes`
  entry from recent `results.jsonl` entries.
- **Never reproduce the same patch** as the previous iteration.
- **Record your reasoning** as a one-line string in `system_prompt_addendum` when the
  failure mode is accuracy-related.

---

## Scoring

The Harness Score is:

```
S = (0.6 × Accuracy) + (0.3 × Format_Adherence) − (0.1 × Token_Latency_Penalty)
```

**Higher is better.** A mutation is **kept** only if `mean_harness_score` strictly improves.
The baseline (version 0) must be run first to establish the reference score.

Secondary metric: `pass_at_n` — fraction of the N trials that reached the correct answer.
Use this to sanity-check the composite score.

---

## The Loop

```
LOOP FOREVER:
  1. Read harness_config.json
  2. Read last 10 results.jsonl entries
  3. Determine best_score (from kept=true entries)
  4. Backup config
  5. Propose ONE strategy patch based on top failure mode
  6. Save mutated config
  7. python aho/harness_runner.py   (runs N parallel trials, outputs JSON to stdout)
  8. Score with eval.py
  9. Log to results.jsonl
 10. Keep if score improved → advance
     Discard if not → git revert harness_config.json (or restore .bak)
 11. Sleep 2s, repeat

NEVER STOP unless manually interrupted (Ctrl+C).
```

---

## Error Recovery Table

| Error type | What happens | Your action |
|---|---|---|
| Mutation proposal fails (JSON parse, timeout) | Researcher skips iteration | Monitor `aho/bug_tracker.jsonl` → simplify prompt |
| `harness_runner.py` crashes | Config reverted automatically | Read bug_tracker for the exception; reduce `n_trials` or `max_steps` if OOM |
| Runner times out (>120s per trial) | Config reverted, logged as crash | Reduce `max_steps` in strategy |
| 5+ consecutive failures | Researcher pauses 60s automatically | Manual investigation recommended |
| Score stagnates for 10+ iterations | No automatic action | Consider resetting to v0 and trying a different `thought_architecture` |

---

## Common Failure Modes → Suggested Fixes

| Failure Mode | Suggested Mutation |
|---|---|
| `json_inside_think_block` | Add `"Never place JSON inside <think>"` to `forbidden_patterns` |
| `hallucinated_tool` | Change `delimiter_style` to `"xml"` |
| `missing_required_tool` | Change `thought_architecture` to `"think_before_every_tool_call"` |
| `accuracy: answer missing keyword(s)` | Change `thought_architecture` to `"reflection_after_tool"` |
| `latency: token_usage is high` | Reduce `max_steps` by 2; add `"Be concise."` to `system_prompt_addendum` |
| `markdown_code_fence_in_tool_args` | Change `tool_call_format` to `"strict_xml"` |
| `tool_parse_error` | Change `delimiter_style` to `"xml"` |
| `missing_required_tool: clothing_suggest` | Add `"You MUST call clothing_suggest after weather_lookup"` to `system_prompt_addendum` |

---

## Output Format (results.jsonl schema)

Each line is a JSON object:
```json
{
  "timestamp": "2026-03-15T23:00:00+00:00",
  "strategy_id": "v3",
  "version": 3,
  "strategy": { ... },
  "pass_at_n": 0.8,
  "mean_harness_score": 0.6320,
  "mean_token_usage": 1240.0,
  "failure_modes": ["accuracy: answer missing keyword(s) (1/5)"],
  "bugs": [],
  "n_bugs": 0,
  "kept": true
}
```

---

## Artifacts Produced (when loop has run ≥ 10 iterations)

| Artifact | How to find |
|---|---|
| **Top-performing prompt** | `python aho/report.py` → "Top Strategies" → strategy #1's `system_prompt_addendum` + `thought_architecture` |
| **Harness logic** | `aho/harness_runner.py` + the winning `strategy` block from `results.jsonl` |
| **Benchmarking report** | `python aho/report.py` (human) or `python aho/report.py --json` (machine) |

---

## Quickstart

```bash
# 1. Check model is running
curl http://localhost:1234/v1/models

# 2. Run baseline (no mutation, just score the default strategy)
python aho/harness_runner.py

# 3. Start autonomous improvement loop
python aho/researcher.py

# 4. Monitor progress (in a separate terminal)
tail -f aho/results.jsonl | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f\"[{r['strategy_id']}] score={r['mean_harness_score']:.4f} kept={r['kept']}\")
"

# 5. Print benchmarking report at any time
python aho/report.py

# 6. View bugs
python aho/report.py --json | python -m json.tool | grep -A3 '"top_bugs"'
```
