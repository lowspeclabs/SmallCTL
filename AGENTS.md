# AGENTS.md

This repository contains SmallCTL, an experimental Python agent harness for small local or self-hosted OpenAI-compatible language models. Its main purpose is to put structure around model-driven technical work: staged task flow, evidence tracking, prompt/context compression, tool dispatch, safety gates, recovery logic, and optional planner/worker/solver style runtimes.

Use this file as the orientation guide for future agents working in the repo. It explains what is where, why the layout exists, and which files usually own a given behavior.

## Fast Start

- Install for development with `pip install -e ".[dev]"`.
- Run the test suite with `pytest`.
- Run the CLI with `smallctl --help` or `python -m smallctl --help`.
- The package entry point is `smallctl = smallctl.main:cli`.
- Primary source lives in `src/smallctl`.
- Primary tests live in `tests`.
- Evaluation fixtures and runner live in `evals` and `scripts/tool_plan_eval.py`.
- The nested `aho` directory is a separate optimizer/evaluation package, not part of the installable `smallctl` package.

## Debug logging

SmallCTL supports granular, structured debug logging:

- `--debug` enables debug logging across all subsystems.
- `--debug-subsystem SUBSYSTEM` enables debug logging for a specific subsystem. Repeatable.
  Valid subsystems: `client`, `graph`, `tools`, `context`, `fama`, `ui`, `memory`, `state`.
- `--debug-tokens` logs every model token; without it, token streams are sampled (first/last 100 + every 20th) to reduce log volume.
- `--log-max-mb N` sets a per-run log-size cap in megabytes (default: 100). When exceeded, the largest channel file is rotated.
- `SMALLCTL_DEBUG_SUBSYSTEMS=client,graph` environment variable works the same as the CLI flag.
- Runtime debug signals: write `escalate:<n>` or `snapshot` to `.smallctl/debug-signal` to escalate logging or dump a recent-event snapshot without restarting.
- Every run directory contains a `run_header.json` sidecar with the event schema version and channel list. Agent-Tools warn when they encounter an unsupported schema version.

When debugging a failed run, start with the decision events:

- `mode_decision` — why a particular runtime mode was chosen.
- `phase_transition` — phase changes and blocked/allowed tools.
- `tool_profile_exposure` / `fama_tool_exposure_applied` — which tools were exposed or hidden.
- `prompt_state_frame_compiled` / `retrieval_selected` — context-frame drops and retrieval rankings.
- `risk_policy_decision` — risk/approval decisions.
- `fama_signal_routed` — FAMA signal → mitigation mapping.

## Repository Map

### Root

- `README.md` explains the product/research intent, install flow, major features, and common CLI invocations.
- `pyproject.toml` defines the `smallctl` package, runtime dependencies, dev dependencies, pytest path, and console script.
- `AGENTS.md` is this orientation file.
- `docs/` contains deeper operational docs. Today it mainly holds write-session procedure notes.
- `scripts/` contains repository-level utility scripts. `scripts/tool_plan_eval.py` is the main eval runner.
- `evals/` contains task fixtures and README files for ToolPlan and test-time-scaling experiments.
- `tests/` contains pytest coverage for harness behavior, graph runtime behavior, tools, UI surfaces, FAMA, ToolPlan, recovery, prompt state, and regressions.
- `src/smallctl/` is the main application package.
- `aho/` is a separate research package for self-improving harness optimization and static/eval tooling.
- `scratch/`, `temp/`, `logs/`, `.smallctl/`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`, virtualenv directories, and `__pycache__` directories are generated or local-working areas. Do not treat them as authoritative source unless the task explicitly points there.
- `phase_plans/`, `refactor_case/`, `Redo/`, and similarly named ad hoc directories are work artifacts or experiments. Read them only when the task is about those plans or cases.

## Core Architecture

SmallCTL is organized around a few explicit layers:

1. CLI and config resolution turn user flags, local config, environment variables, and presets into a `HarnessConfig`.
2. The `Harness` object owns session state, tools, context policy, logging, approvals, and runtime entry points.
3. LangGraph-style runtimes in `src/smallctl/graph` drive model calls, interpretation, tool execution, recovery, and completion.
4. Tool modules in `src/smallctl/tools` provide filesystem, shell, SSH, web, git, memory, artifact, control, and planning capabilities.
5. Context modules in `src/smallctl/context` decide what evidence, memory, artifacts, recent messages, summaries, and recovery hints reach the model.
6. State modules in `src/smallctl/state*.py` and `src/smallctl/state_schema.py` define the durable task/session records that the rest of the system reads and writes.
7. Safety modules such as `risk_policy.py`, `phases.py`, `phase_contracts.py`, `guards.py`, FAMA, loop guards, and write-session code constrain when tools and terminal claims are allowed.
8. UI modules in `src/smallctl/ui` adapt the same harness to the Textual interface.

When making changes, first decide which layer owns the behavior. Most bugs are easier to fix by changing the owning layer instead of patching symptoms in a downstream renderer or test.

## Package Layout

### `src/smallctl/main.py`

This is the CLI entry point. It defines the argparse surface, handles memory subcommands, resolves config, creates logging, builds `HarnessConfig`, and chooses between task execution and the Textual UI.

Look here for:

- CLI flags and subcommands.
- `--run-mode`, `--preset`, `--tui`, `--checkpoint-*`, and model/provider flags.
- How `SmallctlConfig` is projected into `HarnessConfig`.
- Startup/shutdown behavior for CLI runs.

### `src/smallctl/config.py`, `config_support.py`, `config_projection.py`, `presets.py`, `provider_profiles.py`

These files own configuration resolution and provider compatibility.

- `SmallctlConfig` in `config.py` is the broad user-facing config dataclass.
- `resolve_config` merges user config, local `.smallctl.yaml`, `.env`/environment variables, CLI flags, presets, and provider defaults.
- `config_support.py` contains parsing, aliasing, environment, and normalization helpers.
- `config_projection.py` maps `SmallctlConfig` into `HarnessConfig` so CLI and TUI paths stay aligned.
- `presets.py` holds named run profiles such as local coding or small-model-safe defaults.
- `provider_profiles.py` and `src/smallctl/client/adapters/` normalize provider-specific quirks.

When adding a config key, check all relevant dataclasses, projection code, tests such as config parity tests, and any CLI flag exposure.

### `src/smallctl/harness`

The harness package is the central facade and service layer. `src/smallctl/harness/__init__.py` defines `Harness`, then binds behavior from smaller facade modules to avoid one enormous class.

Important files:

- `config.py` defines `HarnessConfig`, the runtime-facing config.
- `initialization.py` constructs state, client, registry, context policy, services, and scratchpad defaults.
- `runtime_facade.py` binds `run_task`, chat/loop/planning/ToolPlan entry points, checkpoint/resume handling, interrupts, teardown, and background persistence.
- `context_facade.py` binds prompt/context assembly helpers.
- `core_facade.py` binds finalization, event emission, logging, result construction, and common lifecycle helpers.
- `task_boundary*.py` manages task resets, handoffs, follow-up classification, remote/local boundary handling, and summaries.
- `tool_dispatch.py`, `tool_dispatch_cache.py`, and `tool_results*.py` connect tool calls to outcomes, artifacts, evidence, verification, memory updates, and postprocessing.
- `tool_result_verification*.py`, `verifier_monitor.py`, and `tool_visibility*.py` own readback/verification of tool effects and which tools/results stay visible.
- `task_classifier*.py`, `task_intent.py`, and `intent_facade.py` classify task intent and route follow-up behavior.
- `approvals.py`, `escalation_*.py`, and `interactive_detector.py` own human approval and escalation behavior.
- `backend_recovery*.py`, `compaction.py`, and `artifact_tracking.py`/`artifact_read_ledger.py` handle backend failure recovery, message compaction, and artifact accounting.
- `reflexion_service.py`, `refine_service.py`, and `trajectory_recorder.py` support repair/refinement/eval traces.
- `subtask_ledger_service.py` and `factory.py` support child harness runs and subtask accounting.

Why this layer exists: graph runtimes should orchestrate, tools should execute, and context modules should render. The harness is the glue that carries state and policy between those layers.

### `src/smallctl/graph`

The graph package contains the execution runtimes and node-level orchestration. These modules are where model outputs become tool calls, tool results, phase transitions, retries, and final outcomes.

Important runtime files:

- `runtime.py` defines `LoopGraphRuntime`, the normal model-tool loop.
- `runtime_chat.py` defines chat-oriented runtime behavior.
- `runtime_planning.py` and `planning_*` files handle planning mode.
- `runtime_tool_plan.py` defines the ToolPlan runtime: planner prompt, JSON plan parsing/validation, optional DAG dispatch, observation compression, solver prompt, and fallback into the normal loop.
- `runtime_indexer.py` defines the code-indexer runtime, and `runtime_specialized.py` re-exports the planning/indexer/tool-plan runtime classes.
- `runtime_staged.py`, `plan_execution.py`, `lifecycle_*`, and `write_session_*` implement staged execution and write-session behavior.
- `test_time_scaling*.py`, `scaling_*`, `solver_refine.py`, and `autocontinue.py` implement candidate generation, retry/scaling helpers, solver refinement, and auto-continue behavior.
- `runtime_auto.py` chooses a runtime when `--run-mode auto` is used.
- `runtime_base.py`, `routing.py`, and `subgraphs.py` contain shared graph compilation, payload serialization, checkpoint config, route helpers, subgraph wiring, and timeout wrappers.

Important flow files:

- `nodes.py`, `lifecycle_nodes.py`, `model_call_nodes.py`, `interpret_nodes.py`, and `tool_execution_nodes.py` are the core graph nodes.
- `model_stream*.py` handles streaming output, parser fallback, halt detection, loop recovery, and provider-related stream issues.
- `tool_call_parser*.py`, `tool_inline_parsing.py`, and `tool_model_rules*.py` parse and normalize tool calls from model output.
- `tool_outcomes*.py`, `task_completion_outcomes.py`, and `terminal_completion.py` resolve tool results and terminal results.
- `tool_loop_guards*.py`, `progress_guard*.py`, `chat_progress*.py`, and `interactive_progress_guard.py` detect lack of progress, loops, repeated reads/writes, and stuck interactive situations.
- `write_recovery*.py`, `tool_execution_recovery*.py`, `tool_artifact_recovery.py`, and `lifecycle_guard_recovery.py` handle failed writes, missing artifacts, schema/tool recovery, and retry nudges.
- `tool_plan_*`, `tool_dag*`, and `rewoo`-related hooks implement planner/worker/solver evidence-gathering paths.
- `cancel_result.py`, `interrupts.py`, and escalation trigger modules handle cancellation and human-interrupt flows.

Why this layer exists: model loops have many state transitions. Keeping them in graph nodes and runtimes makes it possible to test routing, interruptions, fallback, and recovery separately from tool implementation.

### `src/smallctl/tools`

The tools package defines the callable capabilities exposed to the model.

Core concepts:

- `base.py` defines `ToolSpec`, risk labels, mode/profile labels, and the `@tool` helper.
- `registry.py` stores tools and exports OpenAI-compatible function schemas.
- `register.py` builds the registry by collecting tool groups and injecting canonical state/cwd/harness arguments.
- `profiles.py` defines profile names such as `core`, `data`, `network`, `mutate`, `indexer`, and support profiles.

Major tool groups:

- Filesystem: `fs.py`, `fs_listing.py`, `fs_mutations.py`, `fs_patching.py`, `fs_write_flow.py`, `fs_write_sessions.py`, `fs_patch_flow.py`, `ast_patch*.py`.
- Shell/processes: `shell.py`, `shell_foreground.py`, `shell_preflight.py`, `shell_support*.py`, `shell_processes.py`, `process_lifecycle.py`, `process_streams.py`, `local_interactive_sessions.py`.
- SSH/network: `network.py`, `network_ssh_helpers.py`, `network_interactive_sessions.py`, `ssh_files*.py`, `ssh_parsing.py`, remote dispatcher helpers.
- Git: `git_tools.py` and `register_git_tools.py`.
- Web/search: `web.py`, `web_fetch_utils.py`, `web_fetch_artifacts.py`, `web_result_index.py`, `web_budget.py`, `search.py`, `http.py`.
- Memory/artifacts/data: `memory.py`, `artifact.py`, `data.py`, `indexer.py`, `indexer_query.py`.
- Control/planning: `control.py`, `planning.py`, `control_*`, `register_control_planning.py`, `register_operational.py`, `register_filesystem.py`, `register_web_tools.py`, `register_escalation.py`, `register_content.py`.
- Dispatch safety and normalization: `dispatcher*.py`, `type_coerce.py`, `installer_preflight.py`, `network_installer_preflight.py`.

Why this layer exists: tool implementation, schema, risk level, profile, and phase/mode availability are explicit and testable. If a model can call something, it should be registered through this layer.

### `src/smallctl/context`

The context package decides what the model sees. It compiles prompt state from durable state, recent messages, retrieved summaries, artifact snippets, observations, working memory, FAMA capsules, recovery hints, and ReWOO lane frames.

Important files:

- `policy.py` defines context budgets and token-estimation policy.
- `assembler.py`, `assembler_rendering.py`, and `rendering.py` build final message lists.
- `frame.py` defines prompt frame packet structures.
- `frame_compiler.py` is the high-level `PromptStateFrameCompiler`.
- `frame_*_rendering.py` files render phase, run, session, working memory, and recovery sections.
- `frame_invalidation_*` files remove stale facts after verifier failures or conflicting evidence.
- `observations.py` builds compact observation packets from state.
- `retrieval*.py` finds and scores summaries, artifacts, memory, and snippets for prompt inclusion.
- `artifacts.py`, `artifact_visibility.py`, and `artifact_read_coverage.py` decide what artifact content is visible or needs reread.
- `messages*.py` owns recent-message compaction and display of directory/output references.
- `rewoo_lanes.py` builds role-specific planner/solver/refiner frames.
- `step_sandbox.py` supports staged step-local context.
- `subtasks.py` and `tiers.py` render subtask context and tiered episodic summaries.
- `summarizer*.py` handles session/context summarization.

Why this layer exists: small models are sensitive to prompt shape and stale evidence. Prompt compilation is a separate subsystem so the runtime can ask for context without duplicating retrieval and filtering logic.

### `src/smallctl/state.py`, `state_schema.py`, and related state files

State is intentionally explicit and durable.

- `state_schema.py` defines records such as `RunBrief`, `WorkingMemory`, `ExecutionPlan`, `PlanStep`, evidence records, artifacts, claims, context briefs, turn bundles, prompt budget snapshots, write sessions, and experience memory.
- `state.py` defines `LoopState` (which mixes in `LoopStateFlowMixin`), serialization, deserialization, trimming, coercion, migration, and the high-level session state fields used across the harness.
- `state_flow.py` defines `LoopStateFlowMixin`, the bulk of `LoopState`'s flow/behavior methods, and is the largest state module. `state_flow_utils.py` and `state_flow_failure_semantics.py` support its flow helpers and failure semantics.
- `state_coercion.py`, `state_support.py`, `state_records.py`, `state_session_records.py`, and `state_memory.py` keep old checkpoints and loosely structured payloads usable.
- `recovery_schema.py`, `recovery_coercion.py`, and `recovery_metrics.py` define failure, reflexion, subtask, and recovery accounting records.

Why this layer exists: the harness frequently resumes from checkpoints, compresses context, or repairs malformed model/tool output. State schemas and coercion functions are where compatibility and data cleanup belong.

### `src/smallctl/client`

This package wraps OpenAI-compatible chat endpoints.

- `client.py` defines `OpenAICompatClient`, provider profile resolution, stream timeouts, request sizing, and write-heavy request adjustments.
- `client_transport*.py` owns HTTP request construction, client lifecycle, context probing, model metadata, OpenRouter preflight, and llama.cpp repair.
- `streaming.py`, `stream_collectors.py`, and `chunk_parser.py` parse streamed responses and tool-call chunks.
- `provider_adapters.py` and `adapters/` hold provider-specific behavior for generic, OpenRouter, LM Studio, llama.cpp, and similar endpoints.
- `request_budget.py`, `tool_budgeting.py`, and `usage.py` handle context/window accounting.
- `transport_error_classification.py`, `openrouter_preflight.py`, `llamacpp_preflight.py`, and related files turn backend failures into actionable recovery signals.

Why this layer exists: many "OpenAI-compatible" providers differ in stream format, supported parameters, context metadata, timeout behavior, and tool-call streaming. Provider quirks should stay here rather than leaking into graph nodes.

### `src/smallctl/fama`

FAMA is the failure-aware mitigation subsystem. It detects runtime failure modes and injects short mitigation capsules into prompts.

- `signals.py` defines signals from state/tool outcomes.
- `detectors.py`, `detectors_support.py`, and `detector_classifiers.py` classify failures such as loops, early stopping, context drift, stale artifact use, remote/local confusion, and tool-output misreads.
- `capsules.py` renders prompt capsules within a configured token budget.
- `runtime.py`, `router.py`, `state.py`, `config.py`, and `fingerprints.py` manage active mitigations and routing.
- `tool_policy.py` bridges FAMA signals into tool policy.
- `reflexion_bridge.py` connects failure events to reflexion memory.
- `judge.py` supports optional judge-based severity/classification.

Why this layer exists: the harness wants adaptive guardrails without hard-coding every recovery nudge into prompts or graph nodes.

### `src/smallctl/search_server`

This package implements the local web-search/fetch daemon used by web tools.

- `app.py` hosts `/health`, `/search`, and `/fetch` endpoints with token auth.
- `config.py` defines provider/cache/security options.
- `providers.py` and `provider_base.py` abstract search providers.
- `fetch.py`, `extract.py`, and `citations.py` fetch pages, extract text, and produce citation metadata.
- `cache.py` stores positive and negative search/fetch responses.
- `security.py` handles URL and request safety.
- `models.py` defines request/response dataclasses.

Why this layer exists: web work needs caching, citation data, URL safety, and provider abstraction separate from the model loop.

### `src/smallctl/ui`

The UI package is a Textual application over the same harness.

- `app.py` defines `SmallctlApp`, layout, keybindings, startup, shutdown, and high-level UI state.
- `harness_bridge.py` connects harness events to UI rendering.
- `console.py`, `bubbles.py`, `display.py`, `input.py`, and `statusbar.py` render chat, tool/system messages, input, and status.
- `approval.py` and `app_approvals.py` handle human approvals and sudo prompts.
- `app_flow*.py` and `app_actions.py` implement UI commands and task flow.
- `chat_selector.py` and `model_selector.py` support session/model controls.
- `styles.tcss` is the Textual stylesheet.

Why this layer exists: UI should not own harness behavior. It should render events, collect input, and call the harness.

### `src/smallctl/models`

Small dataclasses for shared payloads:

- `conversation.py` defines `ConversationMessage` and retrieval-safe text handling.
- `events.py` defines UI event types and payloads.
- `tool_result.py` defines tool result models.

### Root-level support modules in `src/smallctl`

Many single files at package root are cross-cutting policies or compatibility helpers:

- `phases.py` defines canonical phases and phase-level blocked tools.
- `phase_contracts.py` (with `phase_contracts_support.py`) infers and evaluates explicit/inferred phase contracts.
- `risk_policy.py` determines when tool calls are allowed, blocked, or require approval.
- `evidence.py` normalizes raw tool results into evidence/observation records.
- `plans.py` renders `ExecutionPlan`/`PlanStep` records into text, markdown, and exported plan files.
- `shell_utils.py` provides shell tokenization, command-root/family parsing, and file-read cache keys.
- `tool_output_formatting.py` summarizes and renders structured tool output (for example web search/fetch payloads).
- `client.py` is a legacy compatibility shim re-exporting from the `client/` package.
- `reasoning_policy.py`, `retrieval_safety.py`, `redaction.py`, and `normalization.py` support safe reasoning, retrieval text, secret cleanup, and model-output cleanup.
- `prompts.py`, `prompt_fragments.py`, `prompts_support.py`, and `prompt_model_classifiers.py` hold prompt templates/classifiers outside context-frame rendering.
- `memory_store.py`, `memory_cli.py`, `memory_namespace.py`, and `memory/taxonomy.py` own persistent and CLI-accessible memory.
- `chat_sessions.py` persists resumable chat/session state.
- `logging_utils.py` handles structured run logging.
- `remote_scope.py`, `task_targets.py`, `diagnostic_tasks.py`, `experience_tags.py`, `challenge_progress.py`, and `docker_retry_normalization.py` provide task classification and domain-specific recovery helpers.
- `runtime_error_repair.py`, `guards.py`, `repeat_loop_policy.py`, `interrupt_replies.py`, and `write_session_fsm.py` are targeted policy and recovery helpers.

## Tests

The main tests are under `tests/`. They are mostly focused, behavior-level pytest files rather than broad integration suites.

Common test areas:

- CLI/config: config parity, presets, provider adapters, client import compatibility.
- Graph/runtime: model stream parsing, loop/tool-plan runtimes, staged execution, recursion budgets, cancellation, planning regressions.
- Tools: filesystem, AST patching, shell, SSH, web, git, artifacts, tool schema and visibility.
- Safety/recovery: risk policy, approvals, loop guards, write recovery, missing artifacts, prompt invalidation, backend recovery.
- Context/prompt: prompt state frames, compaction, retrieval, observations, assistant turn layout.
- FAMA/reflexion: detectors, signals, capsules, hard-route behavior, recovery classification.
- UI: app bridge, input bindings, display filters, flow regressions.
- Eval harness: ToolPlan eval runner and related prompt/observation/schema behavior.

Use targeted tests while changing a subsystem, then broaden when touching shared graph, tool dispatch, context, or state code.

## Evals and Research Fixtures

### `evals/tool_plan`

Contains YAML/JSONL tasks and a README for comparing normal loop mode with the ReWOO-style `tool_plan` runtime. The runner records duration, exit code, stdout/stderr, token usage when available, latency metrics, recovery metrics, prompt-shape checks, grounding, fallback behavior, and report decisions.

Typical commands:

```bash
python scripts/tool_plan_eval.py --tasks evals/tool_plan/ --dry-run
python scripts/tool_plan_eval.py --tasks evals/tool_plan/ --mode both --rewoo-frames
```

### `evals/test_time_scaling`

Contains fixtures for staged/test-time-scaling experiments. These are used to evaluate candidate-generation and retry strategies around hard file reads, file mutations, loop status, and runtime probes.

### `aho`

`aho` is a separate package named `aho-optimizer`. Its stated purpose is a self-modifying optimization loop for the SmallCTL harness.

Important areas:

- `aho/main.py`, `harness_runner.py`, `challenge_loop.py`, and `run_baseline.py` run optimizer/eval workflows.
- `aho/researcher.py`, `hypothesis/engine.py`, `fact_extractor.py`, `fact_validator.py`, and `similarity_retriever.py` support hypothesis/research loops.
- `aho/static_analysis/` contains analyzer/config verifier code.
- `aho/validation/` contains probing and validation pipelines.
- `aho/git_manager/` and `git_utils.py` wrap repository operations.
- `aho/knowledge_store/` and `aho/metrics/` store optimizer knowledge/metrics.
- `aho/tests/` contains tests for the nested package.

Do not move `aho` files into `src/smallctl` unless the task explicitly asks for integration. It has its own `pyproject.toml`, dependencies, and package boundaries.

## Runtime Modes and Phases

Canonical phases are defined in `src/smallctl/phases.py`:

- `explore`: gather observations, verify facts, collect open questions; blocks mutation and shell execution.
- `plan`: turn evidence into hypotheses and executable plans; blocks mutation and shell execution.
- `author`: make bounded implementation changes from approved plans; blocks terminal task tools.
- `execute`: run approved actions and verify effects.
- `verify`: compare observed state against expected outcomes; blocks writes and terminal task tools.
- `repair`: recover from failed verifier/execution steps.

Runtime modes include:

- `chat`: conversational runtime.
- `loop`: normal model/tool loop.
- `planning`: plan-oriented mode with approval/interruption behavior.
- `indexer`: code indexer mode.
- `tool_plan`: planner generates bounded read-only evidence plan, workers execute safe reads, observations are compressed, then solver acts.
- `auto`: chooses a runtime based on config/task state.

When debugging "why was this tool blocked?" check phase contracts, tool profiles/modes, registry export filtering, risk policy, dispatcher guards, and graph-level recovery guards.

## Write Sessions

Write sessions are a core reliability feature for staged authoring and large edits. Read `docs/write_session_sop.md` before changing write-session behavior.

Key rules encoded by the harness:

- A `patch_existing` session needs an explicit first write choice: `file_patch`, `ast_patch`, or `file_write(..., replace_strategy="overwrite")`.
- A bare `file_write` to a session-owned path is blocked. The write must include `write_session_id` and `section_name`.
- Session FSM behavior lives in `write_session_fsm.py`, `tools/fs_write_sessions.py`, `tools/fs_write_flow.py`, `tools/fs_write_session_policy.py`, and graph/harness write-session recovery files.
- Tests around chunked writes, write recovery, stranded sessions, and patch-existing regressions are important when touching this area.

## Common Change Guidance

- CLI flag or config change: update `main.py`, `SmallctlConfig`, `HarnessConfig`, `config_projection.py`, any provider/preset defaults, and config parity tests.
- New tool: add implementation in the appropriate `tools/` module, register it through the relevant `register_*` module, assign risk/profile/mode/phase metadata, and add tests for schema, dispatch, and safety.
- Tool dispatch bug: start in `harness/tool_dispatch.py`, `tools/dispatcher*.py`, graph `tool_execution_nodes.py`, and `tool_outcomes*.py`.
- Prompt/context bug: start in `context/frame_compiler.py`, relevant `frame_*_rendering.py`, `context/retrieval*.py`, and prompt state tests.
- Runtime routing bug: start in `graph/runtime*.py`, `runtime_base.py`, `nodes.py`, and route helper tests.
- ToolPlan bug: start in `graph/runtime_tool_plan.py`, `tool_plan_parser.py`, `tool_plan_schema.py`, `tool_plan_safety.py`, `tool_plan_executor.py`, `tool_plan_observations.py`, `tool_dag*.py`, and eval fixtures.
- Model streaming/parser bug: start in `client/streaming.py`, `client/chunk_parser.py`, `graph/model_stream*.py`, and parser tests.
- FAMA mitigation bug: start in `fama/signals.py`, `fama/detectors.py`, `fama/capsules.py`, `fama/router.py`, and FAMA tests.
- Approval/risk bug: start in `risk_policy.py`, `harness/approvals.py`, `harness/escalation_*.py`, `tools/dispatcher_policy_guards.py`, and risk/approval tests.
- UI bug: start in `ui/app.py`, `ui/app_flow*.py`, `ui/harness_bridge.py`, `ui/display.py`, and UI regression tests.
- Search/web bug: start in `search_server/`, `tools/web*.py`, and web tests.
- State/checkpoint compatibility bug: start in `state.py`, `state_schema.py`, `state_coercion.py`, `state_support.py`, and state/schema tests.

## Development Cautions

- The worktree may contain many unrelated local changes. Do not revert or clean files unless explicitly asked.
- Avoid committing generated files from `logs/`, `.smallctl/`, caches, virtualenvs, `temp/`, and `__pycache__`.
- Prefer narrow subsystem edits over broad refactors. Many files are intentionally split to keep graph, harness, context, and tool responsibilities testable.
- Preserve existing recovery behavior when changing happy paths. SmallCTL is mainly valuable because it handles bad tool calls, malformed model output, stale context, and partial writes.
- Be careful with provider compatibility. A change that works for one OpenAI-compatible endpoint may break LM Studio, OpenRouter, llama.cpp, vLLM, or Ollama behavior.
- Treat prompt text changes as behavior changes. Update prompt snapshot or prompt-shape tests when appropriate.
- When changing context retrieval or invalidation, account for stale artifacts and verifier-failure invalidation. Unsupported or optimistic facts should not be reintroduced into prompt state.
- When changing shell, SSH, filesystem, or network tools, check risk labels, approvals, preflight guards, and phase/mode/profile exposure.

## Suggested Verification

For small documentation-only changes:

```bash
python -m py_compile src/smallctl/main.py
```

For source changes, run targeted tests first, for example:

```bash
pytest tests/test_config_parity.py
pytest tests/test_tool_result_verification.py
pytest tests/test_runtime_tool_plan.py
pytest tests/test_write_recovery.py
```

For broad changes touching graph, tools, state, or context:

```bash
pytest
```

For ToolPlan behavior:

```bash
python scripts/tool_plan_eval.py --tasks evals/tool_plan/ --dry-run
```

Use the nested `aho` test suite separately when changing `aho`:

```bash
pytest aho/tests
```
