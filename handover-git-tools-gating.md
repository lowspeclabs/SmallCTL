# Handover: Git Tools Gating Decision

## Context
Phase 6 of the LLMCompiler-lite plan added three new read-only tools to the ToolPlan parallel allowlist:
- `git_status` — `git -C <path> status [--short]`
- `git_diff` — `git -C <path> diff [--cached] [-- <target>]`
- `read_log` — bounded tail read of a workspace log file

These tools are registered with `risk="low"`, `category="git"` (or `"filesystem"` for `read_log`), and `allowed_modes={"chat", "loop", "planning"}`.

## Question: Should git tools require interactive approval gating?

### Assessment: NO

**Rationale:**
1. **Read-only, no mutation risk.** `git_status` and `git_diff` do not modify the working tree, index, or any refs. They are strictly less powerful than `file_read`, which is already ungated and can read any file content in the workspace.
2. **Consistent with existing read-only tools.** Every other tool in `PARALLELIZABLE_TOOL_PLAN_TOOLS` (`file_read`, `dir_list`, `grep`, `find_files`, `artifact_read`, `artifact_grep`, `web_search`, `web_fetch`, `ssh_file_read`) runs without interactive approval. The only approval-gated tool in the codebase is `shell_exec` (`approval_gated_shell`), which executes arbitrary commands.
3. **Latency impact.** Adding interactive approval to DAG-batchable read-only tools would serialise the batch and defeat the latency win that Phase 1–5 were designed to achieve.
4. **Privacy concern is mitigated by scope.** `git_status`/`git_diff` operate on the same workspace that `file_read` and `grep` already have full access to. Any sensitive uncommitted content is already reachable via direct file reads.

### What IS reasonable: a config-level opt-out

Following the existing pattern used for web and artifact tools, a **config flag** lets operators disable git tools in ToolPlan without requiring code changes:

```python
# src/smallctl/config.py
tool_plan_allow_git: bool = True
```

And in `src/smallctl/graph/tool_plan_safety.py`:

```python
if step.tool in {"git_status", "git_diff"} and not allow_git:
    errors.append(f"{step.id}: git tools are disabled for ToolPlan.")
```

This mirrors:
- `tool_plan_allow_web` → gates `web_search`/`web_fetch`
- `tool_plan_allow_artifact_read` → gates `artifact_read`/`artifact_grep`

## Files touched in Phase 6

| File | Change |
|------|--------|
| `src/smallctl/tools/git_tools.py` | New implementations |
| `src/smallctl/tools/register_git_tools.py` | Registration scaffolding |
| `src/smallctl/tools/register.py` | Wired `register_git_tools` into `build_registry` |
| `src/smallctl/graph/tool_plan_schema.py` | Added `git_status`, `git_diff`, `read_log` to `PARALLELIZABLE_TOOL_PLAN_TOOLS` |
| `src/smallctl/graph/tool_plan_safety.py` | Added `read_log`, `git_status`, `git_diff` to `_LOCAL_PATH_TOOLS`; validates `git_diff` `target` arg |
| `tests/test_git_tools.py` | 12 unit tests for the three tools |
| `tests/test_tool_plan_safety.py` | Path-validation tests for git and log tools |
| `tests/test_tool_dag_safety.py` | Allowlist coverage for git tools |
| `tests/test_tool_plan_prompts.py` | Prompt inclusion assertions |
| `src/smallctl/config.py` | `tool_dag_enabled` flipped to `True` |
| `.smallctl.yaml.example` | `tool_dag_enabled` flipped to `true` |

## Implementation (DONE)

A config gate `tool_plan_allow_git: bool = False` has been added following the existing `tool_plan_allow_web` / `tool_plan_allow_artifact_read` pattern:

| File | Change |
|------|--------|
| `src/smallctl/config.py` | Added `tool_plan_allow_git: bool = False` + parsing |
| `.smallctl.yaml.example` | Added `tool_plan_allow_git: false` |
| `src/smallctl/graph/tool_plan_safety.py` | Validates `git_status`/`git_diff` against `allow_git` |
| `src/smallctl/graph/runtime_tool_plan.py` | Reads `tool_plan_allow_git` from config and passes to validator |
| `tests/test_tool_plan_safety.py` | Tests for enabled (`allow_git=True`) and disabled (`allow_git=False`) |

**Default is `False`** — git tools do not appear in validated ToolPlan output unless the user opts in. They remain registered in the harness (so the registry check passes), but validation rejects them at the plan-safety layer when the flag is off.

## Test status
- 116 unit tests pass (`test_tool_plan*.py`, `test_tool_dag*.py`, `test_runtime_tool_plan.py`, `test_git_tools.py`)
- Targeted evals green: `repo_tool_dag_execution`, `wrong_path_recovery_001`, `repo_tool_dispatch_persistence`
