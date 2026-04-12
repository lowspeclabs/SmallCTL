# Large File Refactor Plan

Date: 2026-04-12

## Goal

Reduce the size, duplication, and cross-file coupling in the `smallctl` codebase without changing user-visible behavior.

This plan is intentionally phased so we can:

- preserve behavior while extracting shared logic
- shrink the highest-risk files first
- keep tests green between each step
- avoid one large destabilizing rewrite

## Primary Refactor Targets

These files are the best initial candidates because they are both large and overloaded with multiple responsibilities.

| File | Approx. lines | Why it is a strong candidate |
| --- | ---: | --- |
| `src/smallctl/graph/nodes.py` | 2820 | Mixes initialization, prompt prep, model interpretation, routing, persistence, and recovery behavior |
| `src/smallctl/harness/__init__.py` | 2726 | Acts as orchestrator, classifier, dispatcher, cache owner, artifact manager, and shell-attempt tracker |
| `src/smallctl/graph/tool_call_parser.py` | 2286 | Combines parser rules, model quirks, loop detection, write-session repair, and artifact handling |
| `src/smallctl/graph/tool_outcomes.py` | 2112 | Mixes shell retry hints, write-session verification, recovery nudges, and outcome application |
| `src/smallctl/state.py` | 1962 | Holds state schema, coercion, migration, clipping, and memory helpers |
| `src/smallctl/tools/fs.py` | 1747 | Contains path guards, write-session staging, patch mechanics, repair-cycle tracking, and directory helpers |
| `src/smallctl/graph/model_stream.py` | 1590 | Owns stream classification, recovery, fallback write synthesis, and trace generation |
| `src/smallctl/harness/tool_results.py` | 1140 | Contains result compaction plus duplicated shell/artifact helper logic |
| `src/smallctl/tools/register.py` | 1084 | Large registry builder that is mostly declarative and should be split by profile/category |
| `src/smallctl/tools/shell.py` | 1032 | Process execution, sudo flow, approvals, UI streaming, and lifecycle logic are tightly packed |
| `src/smallctl/graph/runtime.py` | 961 | Similar runtime shapes repeated across loop/chat/planning/indexer variants |
| `src/smallctl/context/retrieval.py` | 883 | Retrieval pipeline plus artifact visibility policy and query-building logic |
| `src/smallctl/ui/app.py` | 865 | Main app responsibilities are broad enough to split event wiring from view logic |
| `src/smallctl/client/client.py` | 843 | Client, timeout policy, and timeline collection are still tightly coupled |
| `src/smallctl/context/assembler.py` | 783 | Prompt assembly plus prompt visibility rules and clipping heuristics live together |
| `src/smallctl/tools/network.py` | 641 | SSH command construction and subprocess streaming overlap with shell-process logic |
| `src/smallctl/context/messages.py` | 578 | Large enough to review after higher-priority extractions land |
| `src/smallctl/config.py` | 547 | Parsing, normalization, env loading, and profile application can be separated |
| `src/smallctl/graph/checkpoint.py` | 545 | Checkpoint storage and helpers may benefit from modularization |

## Known Duplication To Eliminate First

These are the clearest centralization wins.

### 1. Run-mode classification logic

Duplicated or near-duplicated logic exists across:

- `src/smallctl/harness/__init__.py`
- `src/smallctl/harness/run_mode.py`

Examples:

- `_is_smalltalk`
- `_needs_loop_for_content_lookup`
- `_needs_contextual_loop_escalation`
- `_looks_like_execution_followup`
- `_looks_like_action_request`
- `_needs_memory_persistence`
- `_looks_like_shell_request`

Target extraction:

- `src/smallctl/harness/task_classifier.py`

### 2. Shell attempt family and command-normalization helpers

Duplicated helpers exist across:

- `src/smallctl/harness/__init__.py`
- `src/smallctl/harness/tool_results.py`
- `src/smallctl/tools/shell.py`
- `src/smallctl/risk_policy.py`

Examples:

- `_shell_tokens`
- `_looks_like_env_assignment`
- `_file_read_cache_key`
- `_shell_attempt_family_key`
- `_shell_attempt_is_diagnostic`
- `_shell_command_root`
- `_shell_unwrap_command`

Target extraction:

- `src/smallctl/shell_utils.py` or `src/smallctl/harness/shell_families.py`

### 3. Artifact visibility policy

Duplicated helpers exist across:

- `src/smallctl/context/assembler.py`
- `src/smallctl/context/retrieval.py`

Examples:

- `_is_superseded_artifact`
- `_is_prompt_visible_artifact`

Target extraction:

- `src/smallctl/context/artifact_visibility.py`

### 4. Payload coercion and state normalization helpers

Similar coercion logic exists across:

- `src/smallctl/state.py`
- `src/smallctl/graph/state.py`

Target extraction:

- `src/smallctl/state/coercion.py` or `src/smallctl/state_helpers.py`

### 5. Subprocess stream-reading and event emission

Very similar stream-reading code exists across:

- `src/smallctl/tools/shell.py`
- `src/smallctl/tools/network.py`

Target extraction:

- `src/smallctl/tools/process_streams.py`

### 6. Provider adapter boilerplate

Mostly repetitive adapters exist across:

- `src/smallctl/client/adapters/generic.py`
- `src/smallctl/client/adapters/lmstudio.py`
- `src/smallctl/client/adapters/openrouter.py`

Target extraction:

- shared base adapter or a config-driven adapter factory

### 7. UI modal boilerplate

Repeated button-focus and modal handling exists in:

- `src/smallctl/ui/approval.py`

Target extraction:

- `src/smallctl/ui/modal_base.py`

### 8. Repeated test scaffolding

Repeated fake registry, fake process, and fake emitter patterns appear in:

- `tests/test_plan_playbook.py`
- `tests/test_backend_unload_recovery.py`
- `tests/test_compaction_progress.py`
- `tests/test_qwen_parser_and_ssh.py`

Target extraction:

- `tests/helpers/` fixtures and fake objects

## Constraints

This refactor should follow these rules:

- no behavior changes unless explicitly called out
- no large cross-cutting rename-only churn unless it unlocks a split
- keep public tool names and runtime semantics stable
- preserve existing tests before optimizing structure further
- prefer extraction plus delegation over rewriting logic from scratch

## Phase 0: Baseline And Safety Rails

### Objective

Create a stable baseline before moving logic.

### Work

- Capture file-size baseline for all `src/**/*.py` files over 500 lines
- Tag the current duplication seams listed above
- Add or tighten tests around the highest-risk extracted logic before moving it
- Document expected invariants for:
  - run-mode selection
  - shell command normalization
  - write-session fallback behavior
  - tool-call parsing
  - artifact visibility

### Suggested outputs

- this refactor plan
- optional temporary `temp/refactor_inventory.py` or one-off command notes
- missing focused tests for extracted helpers

### Exit criteria

- current large-file list is recorded
- high-risk helper behavior is covered by tests
- no production code moved yet

### Validation

```bash
pytest
```

## Phase 1: Extract Shared Utility Modules

### Objective

Remove exact or near-exact duplication before splitting the largest files.

### Work

- Extract run-mode classification helpers from `harness/__init__.py` and `harness/run_mode.py`
- Extract shell attempt and command-token helpers from harness/tool-results/shell/risk-policy code
- Extract artifact visibility helpers used by prompt assembly and retrieval
- Extract shared subprocess stream-reading helpers for `shell.py` and `network.py`
- Extract repeated state payload coercion helpers into a shared module

### Files likely created

- `src/smallctl/harness/task_classifier.py`
- `src/smallctl/harness/shell_attempts.py`
- `src/smallctl/context/artifact_visibility.py`
- `src/smallctl/tools/process_streams.py`
- `src/smallctl/state_helpers.py` or `src/smallctl/state/coercion.py`

### Expected payoff

- immediate line-count reduction in multiple files
- one source of truth for duplicated behavior
- lower risk for later file splits

### Exit criteria

- duplicated helper bodies removed from original files
- imports point to shared modules
- tests confirm no behavior drift

### Validation

```bash
pytest
pytest tests/test_qwen_parser_and_ssh.py
pytest tests/test_compaction_progress.py
```

## Phase 2: Break Up `Harness`

### Objective

Turn `src/smallctl/harness/__init__.py` from a “god object” into a thin coordinator.

### Work

- Keep `Harness` as the public façade
- Move task classification and memory intent helpers out first
- Move tool-dispatch support helpers into a dedicated module
- Move artifact/shell-attempt bookkeeping into dedicated helpers
- Move chat-mode tool exposure logic into a separate policy module

### Proposed module split

- `src/smallctl/harness/task_classifier.py`
- `src/smallctl/harness/tool_dispatch.py`
- `src/smallctl/harness/artifact_tracking.py`
- `src/smallctl/harness/chat_tool_policy.py`
- `src/smallctl/harness/conversation_logging.py`

### Specific code to target

- `_extract_intent_state`
- `_infer_environment_tags`
- `_infer_entity_tags`
- `_infer_requested_tool_name`
- `_chat_mode_tools`
- `_dispatch_tool_call`
- `_file_read_cache_key`
- shell-attempt family helpers

### Exit criteria

- `harness/__init__.py` mainly wires dependencies and delegates
- no duplicate copies remain in `harness/tool_results.py`
- file size materially reduced

### Validation

```bash
pytest tests/test_plan_playbook.py
pytest tests/test_backend_unload_recovery.py
pytest tests/test_compaction_progress.py
```

## Phase 3: Split Graph Runtime And Node Logic

### Objective

Separate runtime orchestration from node behavior and mode-specific logic.

### Work

- In `graph/runtime.py`, extract shared runtime base behavior from mode-specific variants
- Replace repeated `run` and payload bootstrap patterns with a reusable base implementation
- Split `graph/nodes.py` by domain
- Keep current graph edges and node names stable during the split

### Proposed module split

- `src/smallctl/graph/nodes/init_nodes.py`
- `src/smallctl/graph/nodes/prompt_nodes.py`
- `src/smallctl/graph/nodes/model_nodes.py`
- `src/smallctl/graph/nodes/interpret_nodes.py`
- `src/smallctl/graph/nodes/tool_nodes.py`
- `src/smallctl/graph/runtime_base.py`

### Specific code to target

- initialize/resume nodes
- prompt-prep nodes
- model-call and interpret nodes
- dispatch/persist nodes
- mode-specific route helpers

### Exit criteria

- `graph/runtime.py` becomes a small composition layer
- `graph/nodes.py` is replaced by smaller domain modules or reduced to exports
- no graph-contract changes visible to callers

### Validation

```bash
pytest tests/test_phase_contracts.py
pytest tests/test_small_model_freeze_guard.py
pytest tests/test_streaming_halt_detection.py
pytest tests/test_plan_playbook.py
```

## Phase 4: Split Tool Parsing And Tool Outcomes

### Objective

Separate parsing, guards, recovery, and outcome application into coherent modules.

### Work on `graph/tool_call_parser.py`

- move model-specific parsing quirks into provider/model rule modules
- move repeated-tool-loop detection into a guard module
- move write-session repair logic into a write-session parser support module
- move artifact-read recovery into artifact guard helpers

### Work on `graph/tool_outcomes.py`

- split shell outcome handling from write-session outcome handling
- split verifier/repair nudges from generic tool outcome application
- isolate UI-event emission helpers

### Proposed module split

- `src/smallctl/graph/tool_parsing/model_rules.py`
- `src/smallctl/graph/tool_parsing/repeat_guards.py`
- `src/smallctl/graph/tool_parsing/write_session_repair.py`
- `src/smallctl/graph/tool_parsing/artifact_guards.py`
- `src/smallctl/graph/tool_outcomes/shell_outcomes.py`
- `src/smallctl/graph/tool_outcomes/write_session_outcomes.py`
- `src/smallctl/graph/tool_outcomes/chat_progress.py`

### Exit criteria

- parser file no longer mixes five unrelated concerns
- outcome file no longer mixes all recovery policies
- each new module has targeted tests

### Validation

```bash
pytest tests/test_plan_playbook.py
pytest tests/test_write_recovery.py
pytest tests/test_write_recovery_regression.py
pytest tests/test_phase_contracts.py
```

## Phase 5: Split State And Filesystem Authoring Support

### Objective

Make state and file-authoring behavior easier to understand and safer to change.

### Work on `state.py`

- separate dataclasses/schema definitions from coercion functions
- separate migration and serialization helpers
- keep `LoopState` import path stable if possible

### Proposed split

- `src/smallctl/state_schema.py`
- `src/smallctl/state_coercion.py`
- `src/smallctl/state_migrations.py`
- `src/smallctl/state_serialization.py`

### Work on `tools/fs.py`

- isolate write-session path/session helpers
- isolate exact patching helpers
- isolate repair-cycle tracking
- isolate read/list/tree helpers from mutating operations

### Proposed split

- `src/smallctl/tools/fs_sessions.py`
- `src/smallctl/tools/fs_patching.py`
- `src/smallctl/tools/fs_guards.py`
- `src/smallctl/tools/fs_listing.py`

### Exit criteria

- state schema is readable without scrolling through coercion code
- filesystem tool behavior is grouped by concern
- authoring and read-only helpers are clearly separated

### Validation

```bash
pytest tests/test_write_session_state_machine.py
pytest tests/test_write_recovery.py
pytest tests/test_dir_list_display.py
pytest tests/test_state_schema_versioning.py
```

## Phase 6: Simplify Tools, Client, UI, And Registry

### Objective

Finish the medium-sized structural cleanup after the largest risk areas are under control.

### Work

- split `tools/register.py` by tool category or profile
- extract shared process streaming from `tools/shell.py` and `tools/network.py`
- reduce adapter boilerplate under `client/adapters/`
- split `ui/app.py` into event bridge, action handlers, and composition/view helpers
- split `config.py` into loaders, normalizers, and profile application helpers

### Proposed module split

- `src/smallctl/tools/register_filesystem.py`
- `src/smallctl/tools/register_control.py`
- `src/smallctl/tools/register_network.py`
- `src/smallctl/tools/process_streams.py`
- `src/smallctl/ui/app_actions.py`
- `src/smallctl/ui/app_events.py`
- `src/smallctl/config_loader.py`
- `src/smallctl/config_profiles.py`

### Exit criteria

- registry builder is not a single thousand-line function
- shell and network subprocess handling share the same core implementation
- UI modal and app flow are easier to trace

### Validation

```bash
pytest tests/test_client_health_check.py
pytest tests/test_client_timeouts.py
pytest tests/test_provider_adapter_registry.py
pytest tests/test_qwen_parser_and_ssh.py
pytest tests/test_tool_result_display.py
```

## Phase 7: Test Refactor And Enforcement

### Objective

Prevent the same large-file drift from reappearing.

### Work

- extract repeated test doubles into `tests/helpers/`
- reduce `tests/test_plan_playbook.py` by moving fixtures and fake classes out
- optionally add a lightweight size/duplication check in CI

### Suggested helper modules

- `tests/helpers/fake_registry.py`
- `tests/helpers/fake_process.py`
- `tests/helpers/fake_harness.py`
- `tests/helpers/fake_emitters.py`

### Optional guardrails

- warn when a source file exceeds 1000 lines
- warn when exact duplicate helper bodies are introduced in key modules

### Exit criteria

- biggest tests rely on shared fixtures
- future structural regressions are easier to spot

### Validation

```bash
pytest
```

## Recommended Execution Order

This is the safest sequence.

1. Phase 0 baseline and missing tests
2. Phase 1 shared utility extraction
3. Phase 2 harness decomposition
4. Phase 3 runtime and node split
5. Phase 4 parser and outcome split
6. Phase 5 state and filesystem split
7. Phase 6 tools/client/UI/config cleanup
8. Phase 7 test cleanup and guardrails

## Success Metrics

The refactor is succeeding if we see most of the following:

- no single core source file remains above roughly 1200 to 1500 lines
- duplicated helper implementations are removed from harness, shell, and context layers
- runtime behavior is unchanged under the current test suite
- core modules read as one concern each
- new contributors can find classification, parsing, state coercion, and process-running logic without hunting across unrelated files

## Non-Goals

These should not be bundled into the refactor unless needed by a specific phase.

- changing public tool names
- redesigning the runtime architecture from scratch
- changing planner/loop semantics
- rewriting tests to a new framework
- broad naming churn without structural payoff

## First Concrete Slice

If we want the best low-risk starting point, begin here:

1. Extract run-mode classification helpers from `harness/__init__.py` and `harness/run_mode.py`
2. Extract shell attempt family helpers from `harness/__init__.py` and `harness/tool_results.py`
3. Extract artifact visibility helpers from `context/assembler.py` and `context/retrieval.py`
4. Add shared test fixtures for fake registry and fake process patterns

That slice should reduce duplication immediately, trim several large files, and make the later decompositions much safer.
