# RCH ToolPlan/ReWOO Refinement Plan

## Goal

Refine the current ToolPlan/ReWOO path so planner, solver, and bounded-refine prompts receive small role-specific context frames instead of inheriting the normal mixed prompt surface.

The goal is not to replace the existing context stack. Keep `PromptStateFrame`, `PromptStateFrameCompiler`, `PromptAssembler`, `ReasoningGraph`, `WorkingMemory`, `ExperienceMemory`, turn bundles, context briefs, FAMA capsules, and artifact records as the storage and general prompt substrate. Add a narrow compiler that reads those structures and renders ToolPlan-specific lanes.

## Current Codebase Shape

Structured context:
- `src/smallctl/context/frame.py` defines `PromptStateFrame`, `PromptStateSpine`, `PromptStateDrop`, evidence, experience, artifact, and phase packets.
- `src/smallctl/context/frame_compiler.py` compiles `LoopState` into `PromptStateFrame`.
- `src/smallctl/context/assembler.py` builds normal model prompts from the frame, recent transcript, warm briefs, summaries, observations, artifacts, and warm memories.
- `src/smallctl/context/observations.py` converts `ReasoningGraph.evidence_records` into `ObservationPacket`s.
- `src/smallctl/context/summarizer.py` already has observation-first turn bundle compaction, with generic summary fallback paths.
- `src/smallctl/context/policy.py` provides `ContextPolicy` and `estimate_text_tokens()`.

Durable state:
- `src/smallctl/state_schema.py` defines `RunBrief`, `WorkingMemory`, `ExecutionPlan`, `PlanStep`, `EvidenceRecord`, `DecisionRecord`, `ReasoningGraph`, and `ExperienceMemory`.
- `src/smallctl/state.py` defines `LoopState`, including `run_brief`, `working_memory`, `reasoning_graph`, `active_plan`, `warm_experiences`, `failure_events`, `scratchpad`, artifacts, summaries, briefs, and turn bundles.
- `src/smallctl/state_records.py`, `state_memory.py`, and `state_coercion.py` handle coercion and rendering support for those records.

ToolPlan runtime:
- `src/smallctl/graph/runtime_tool_plan.py` is the main lifecycle: initialize -> planner prompt -> planner model call -> parse/validate -> dispatch -> persist -> compress observations -> solver prompt -> solver model call -> interpret.
- `_prepare_planner_prompt()` currently builds a task-only prompt with `build_tool_plan_planner_prompt(task=..., max_steps=...)`, plus a repair nudge from `scratchpad["_tool_plan_repair_nudge"]`.
- `_prepare_solver_prompt()` currently appends `build_tool_plan_solver_system_suffix(observations_text, ...)` to recent messages and then calls generic `prepare_prompt()`.
- `_compress_observations_node()` builds `ToolPlanObservation` records, renders `TOOL PLAN OBSERVATIONS`, stores `_tool_plan_observations_text`, updates ToolPlan metrics, and attaches a compact string to the subtask ledger via `_attach_tool_plan_evidence()`.
- `tool_dag_enabled` already provides optional parallel read-only dispatch through `src/smallctl/graph/tool_dag.py`, `tool_dag_executor.py`, and `tool_dag_safety.py`.
- `solver_refine_enabled` already provides bounded critique through `src/smallctl/harness/refine_service.py` and `src/smallctl/graph/solver_refine.py`.
- `src/smallctl/harness/trajectory_recorder.py` records ToolPlan trajectories using `_tool_plan`, `_tool_plan_observations_text`, `_last_solver_draft`, and refine metadata.

ToolPlan support:
- `src/smallctl/graph/tool_plan_prompts.py` owns planner and solver prompt text.
- `src/smallctl/graph/tool_plan_observations.py` defines `ToolPlanObservation`, `build_tool_plan_observations()`, and `render_tool_plan_observations()`.
- `src/smallctl/graph/tool_plan_schema.py`, `tool_plan_parser.py`, and `tool_plan_safety.py` define, parse, and validate read-only evidence plans.
- `src/smallctl/graph/tool_plan_executor.py` translates a ToolPlan into pending tool calls.

Configuration:
- `src/smallctl/config.py`, `config_support.py`, `main.py`, and `harness/bootstrap_support.py` already plumb ToolPlan flags such as `tool_plan_runtime_enabled`, `tool_plan_auto_select`, observation limits, web/artifact permissions, DAG flags, and solver-refine flags.
- New ReWOO lane flags should follow that existing config plumbing pattern.

## Implementation Principles

- Add a sibling ToolPlan role compiler; do not destabilize `PromptAssembler` for normal chat, loop, planning, or indexer prompts.
- Reuse `PromptStateFrameCompiler` as the first extraction layer where possible, then filter/render by role.
- Keep generic transcript summaries as fallback/audit context for ToolPlan, not the default planner or solver substrate.
- Prefer deterministic pure compile/render functions. Runtime wiring should be thin and easy to disable.
- Preserve legacy `_tool_plan_observations_text` until the lane path is the default, because current refine, trajectory recording, metrics, and tests use it.
- Keep all rollout behind config flags before changing defaults.
- Treat this as a prompt-shape and evidence-normalization change, not a broad RCH rewrite.

## Target Lanes

### Plan State

Sources:
- `LoopState.run_brief`
- `LoopState.working_memory`
- `LoopState.active_plan` and `ExecutionPlan.active_step()`
- `LoopState.active_step_id`
- `LoopState.failure_events`
- `LoopState.scratchpad["_tool_plan_repair_nudge"]`
- current `ToolPlan` in `scratchpad["_tool_plan"]` when present
- subtask ledger summary when available through `harness.subtask_ledger`

Fields:
- goal and task contract
- constraints and acceptance criteria
- active plan and active step
- pending dependencies
- prior failed approaches
- open questions
- repair/retry nudges
- relevant FAMA/subtask hard-route reasons when already captured in state

### Normalized Evidence

Sources:
- `ReasoningGraph.evidence_records`
- `context.observations.build_observation_packets()`
- `ToolPlanObservation` records from the current dispatch
- artifact ids and operation ids attached to tool results
- `_tool_plan_evidence_ids` once ToolPlan observations are imported into `ReasoningGraph`

Fields:
- evidence id
- statement or observation summary
- source tool/path/query/artifact
- success/failure or negative flag
- confidence/freshness
- duplicate-of and related refs
- operation id and artifact id

### Decisions

Sources:
- `ReasoningGraph.decision_records`
- `WorkingMemory.decisions`
- active/superseded metadata in `DecisionRecord`

Fields:
- decision id
- decision text or rationale
- evidence refs
- status: active, superseded, rejected
- phase, plan step, requested tool, and intent when present

### Reusable Experiences

Sources:
- retrieved `ExperienceMemory`
- `LoopState.warm_experiences`
- existing lexical retrieval in `src/smallctl/context/retrieval.py`

Fields:
- lesson or notes
- applies-when tags
- avoid/action guidance from outcome/failure fields
- evidence refs
- confidence and reuse count

## New Module: `src/smallctl/context/rewoo_lanes.py`

Add a small compiler module. It should read the current state shape, not introduce a parallel state system.

```python
ReWOORole = Literal["planner", "solver", "refiner"]

@dataclass(slots=True)
class ReWOOPlanLane:
    goal: str = ""
    task_contract: str = ""
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    active_plan: list[str] = field(default_factory=list)
    active_step: str = ""
    pending_dependencies: list[str] = field(default_factory=list)
    prior_failures: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    repair_nudges: list[str] = field(default_factory=list)

@dataclass(slots=True)
class ReWOOEvidenceLane:
    packets: list[ObservationPacket] = field(default_factory=list)
    tool_plan_observations: list[ToolPlanObservation] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

@dataclass(slots=True)
class ReWOODecisionLane:
    records: list[DecisionRecord] = field(default_factory=list)
    memory_decisions: list[str] = field(default_factory=list)

@dataclass(slots=True)
class ReWOOExperienceLane:
    memories: list[ExperienceMemory] = field(default_factory=list)

@dataclass(slots=True)
class ReWOOLaneFrame:
    role: ReWOORole
    plan: ReWOOPlanLane = field(default_factory=ReWOOPlanLane)
    evidence: ReWOOEvidenceLane = field(default_factory=ReWOOEvidenceLane)
    decisions: ReWOODecisionLane = field(default_factory=ReWOODecisionLane)
    experiences: ReWOOExperienceLane = field(default_factory=ReWOOExperienceLane)
    drop_log: list[PromptStateDrop] = field(default_factory=list)
```

Compiler API:

```python
class ReWOOLaneCompiler:
    def __init__(self, policy: ContextPolicy | None = None) -> None: ...

    def compile(
        self,
        *,
        state: LoopState,
        role: ReWOORole,
        retrieved_experiences: Iterable[ExperienceMemory] = (),
        tool_plan_observations: Iterable[ToolPlanObservation] = (),
        token_budget: int | None = None,
    ) -> ReWOOLaneFrame: ...

    def render(self, frame: ReWOOLaneFrame, *, token_budget: int | None = None) -> str: ...
```

Rendering sections:
- `REWOO PLAN STATE`
- `REWOO EVIDENCE`
- `REWOO DECISIONS`
- `REWOO EXPERIENCES`
- `REWOO ROLE RULES`

Budgeting:
- Use `estimate_text_tokens()`.
- Emit `PromptStateDrop(lane=f"rewoo.{role}.{lane}", reason=...)` whenever items are dropped.
- Preserve newest/current-dispatch evidence first, then active decisions, then high-confidence experiences.

Exports:
- Add `ReWOOLaneCompiler`, `ReWOOLaneFrame`, and `ReWOORole` to `src/smallctl/context/__init__.py`.

## Role Frames

### Planner Frame

Include:
- goal, task contract, constraints, and acceptance criteria
- current plan state from `working_memory.plan`, `run_brief.implementation_plan`, and `active_plan.compact_lines()`
- prior failures from `working_memory.failures` and recent `failure_events`
- open questions from `working_memory.open_questions`
- repair nudge from `_tool_plan_repair_nudge`
- a small set of retrieved or warm experiences
- optional evidence index only when it mentions current files, blockers, or failed approaches

Exclude:
- full transcript
- raw tool outputs
- generic warm briefs unless reduced to failure/question/evidence ids
- artifact snippets
- rendered `TOOL PLAN OBSERVATIONS`

Budget:
- plan state: 40%
- failures/open questions: 25%
- experiences: 20%
- evidence index: 15%

### Solver Frame

Include:
- goal and acceptance criteria
- normalized `ToolPlanObservation` records from the current dispatch
- current ToolPlan evidence ids after import into `ReasoningGraph`
- recent relevant `ObservationPacket`s
- unresolved contradictions and evidence gaps
- active decisions and memory decisions
- minimal action affordances from `build_tool_plan_solver_system_suffix()`

Exclude:
- generic warm summaries by default
- unrelated retrieved experiences
- raw transcript except latest user task if absent from the goal
- fresh tool output blocks already represented by evidence packets

Budget:
- evidence packets and current ToolPlan observations: 55%
- contradictions/gaps: 20%
- decisions: 15%
- acceptance criteria/action affordances: 10%

### Refiner Frame

Use this for `RefineService.run_bounded_refine()` after planner/solver lanes are stable.

Include:
- solver draft
- acceptance checklist
- ToolPlan evidence refs
- failed observations
- contradictions/gaps
- verifier signals when available

Budget:
- draft/check target: 35%
- acceptance criteria: 25%
- evidence refs: 25%
- blockers/contradictions: 15%

## Phase 1: Add Compiler Without Runtime Behavior Change

1. Create `src/smallctl/context/rewoo_lanes.py`.
2. Export the compiler and frame types from `src/smallctl/context/__init__.py`.
3. Implement extraction helpers:
   - `_compile_plan_lane(state)`
   - `_compile_evidence_lane(state, role, tool_plan_observations)`
   - `_compile_decision_lane(state, role)`
   - `_compile_experience_lane(state, role, retrieved_experiences)`
4. Render stable sections listed above.
5. Add config fields and env/plumbing:
   - `rewoo_lane_frames_enabled: bool = False`
   - `rewoo_planner_frame_enabled: bool = False`
   - `rewoo_solver_frame_enabled: bool = False`
   - `rewoo_refiner_frame_enabled: bool = False`
   - `rewoo_frame_token_budget: int = 1200`
6. Wire the new config fields through `src/smallctl/config.py`, `config_support.py`, `main.py`, and `harness/bootstrap_support.py`.

Tests:
- Add `tests/test_rewoo_lanes.py`.
- Assert planner/solver sections render from synthetic `LoopState`.
- Assert drop logs include `rewoo.<role>.<lane>` lane names.
- Assert disabled flags do not alter existing ToolPlan planner or solver prompts.

## Phase 2: Wire Planner Frame

Change `_prepare_planner_prompt()` in `src/smallctl/graph/runtime_tool_plan.py`.

Current behavior:
- Build `build_tool_plan_planner_prompt(task=task, max_steps=max_steps)`.
- Pop `_tool_plan_repair_nudge` and append it as repair text.

New behavior behind `rewoo_planner_frame_enabled`:
- Peek at `_tool_plan_repair_nudge` before popping it so the lane compiler can include it.
- Compile `ReWOOLaneCompiler.compile(state=harness.state, role="planner", token_budget=...)`.
- Render the frame.
- Extend `build_tool_plan_planner_prompt()` with optional `context_frame: str = ""`.
- Preserve the old task-only prompt when the flag is disabled or compile fails.

Prompt rule:
- The planner may use the frame to choose reads, but must still return only ToolPlan JSON.
- The frame must not include recent assistant/tool transcript blobs.

Tests:
- Update `tests/test_runtime_tool_plan.py`.
- Capture planner messages and assert they include `REWOO PLAN STATE` when enabled.
- Assert prior failures/open questions/relevant experiences appear.
- Assert raw tool chatter and `TOOL PLAN OBSERVATIONS` do not appear in planner messages.
- Assert repair nudge is included once and still asks for corrected JSON.

## Phase 3: Normalize ToolPlan Observations Into ReasoningGraph Evidence

Add conversion helpers in `src/smallctl/graph/tool_plan_observations.py`:

```python
def observation_to_evidence_record(
    observation: ToolPlanObservation,
    *,
    objective: str,
    step_index: int,
    created_at_step: int,
) -> EvidenceRecord: ...

def attach_tool_plan_observation_evidence(
    state: LoopState,
    *,
    objective: str,
    observations: list[ToolPlanObservation],
) -> list[str]: ...
```

Record mapping:
- `evidence_id`: deterministic, for example `TP-E{created_at_step}-{step_id}`
- `kind`: `"tool_plan_observation"` for success, `"tool_plan_negative_observation"` for failure
- `statement`: success summary or failure error
- `phase`: `"tool_plan"`
- `tool_name`: observation tool
- `operation_id`, `artifact_id`: copy through
- `source`: path, query, or `"tool_plan:<step_id>"`
- `confidence`: `0.8` for success, `0.4` for failure/missing record
- `negative`: `not observation.success`
- `metadata`: include `tool_plan_step_id`, `path`, `query`, `duplicate_of`, `objective`, and `observation_adapter="tool_plan_observation"`

Wire point:
- In `_compress_observations_node()`, after `build_tool_plan_observations()` and before rendering/stashing `_tool_plan_observations_text`, attach evidence records and stash returned ids in `scratchpad["_tool_plan_evidence_ids"]`.
- Also stash the structured observations in `scratchpad["_tool_plan_observations"]` for solver lane compilation.
- Keep `_attach_tool_plan_evidence()` for subtask-ledger continuity until the new evidence path is fully adopted.

Tests:
- Add or update `tests/test_tool_plan_observations.py`.
- Assert conversion preserves artifact/operation/path/query/error fields.
- Assert duplicate observations reference `duplicate_of`.
- Assert repeated attachment does not duplicate evidence ids.
- Update `tests/test_evidence_normalization.py` so `build_observation_packets()` classifies `tool_plan_observation` packets cleanly.

## Phase 4: Wire Solver Frame

Change `_prepare_solver_prompt()` in `src/smallctl/graph/runtime_tool_plan.py`.

Current behavior:
- Pop `_tool_plan_observations_text`.
- Append a system message containing freeform observations.
- Call generic `prepare_prompt()`.

New behavior behind `rewoo_solver_frame_enabled`:
- Read structured observations from `_tool_plan_observations`; fall back to `_tool_plan_observations_text` only for compatibility.
- Compile `ReWOOLaneCompiler.compile(role="solver", tool_plan_observations=..., token_budget=...)`.
- Render the frame and pass it to `build_tool_plan_solver_system_suffix(frame_text, ...)`.
- Return a narrow message list directly instead of calling generic `prepare_prompt()`.
- Include the latest visible user task as a user message if the frame does not already contain the goal.

Compatibility path:
- If the flag is disabled or the compiler raises, keep the current append-and-`prepare_prompt()` behavior.
- Preserve `_tool_plan_observations_text` long enough for refine, trajectory recording, metrics, and existing tests.

Tests:
- Update `tests/test_runtime_tool_plan.py`.
- Assert solver prompt includes `REWOO EVIDENCE`.
- Assert solver prompt includes current ToolPlan evidence ids.
- Assert solver prompt does not include unrelated warm summaries or generic retrieved knowledge sections.
- Assert solver prompt still contains action affordances and fresh output limit.
- Assert token metrics still record `tool_plan_solver_tokens`.

## Phase 5: Refiner Frame

Current behavior:
- `RefineService.run_bounded_refine()` builds a prompt with raw `observations_text`, draft, optional active subtask, and verifier signals.
- `runtime_tool_plan.py` calls refine after the solver draft when `solver_refine_enabled` is true.

New behavior behind `rewoo_refiner_frame_enabled`:
- Compile `ReWOOLaneCompiler.compile(role="refiner", tool_plan_observations=..., token_budget=...)`.
- Extend `build_critique_prompt()` or add a sibling builder that accepts `context_frame`.
- Prefer structured evidence and failed observations over raw `TOOL PLAN OBSERVATIONS`.
- Keep raw observations fallback when frame compilation exceeds `solver_refine_token_budget`.

Tests:
- Update `tests/test_solver_refine.py` and `tests/test_solver_refine_prompt_budget.py`.
- Assert the refiner sees failed observations and acceptance criteria.
- Assert oversized frames still fall back or skip according to existing budget behavior.

## Phase 6: Lane-Aware Compaction

Keep the current tiered compaction system, but make normal structured compaction update lane-shaped metadata rather than depending on generic episodic summaries for ToolPlan.

Changes:
- Add a pure extraction helper in `src/smallctl/context/summarizer.py`, for example `extract_rewoo_lanes_from_messages(state, messages, observation_packets)`.
- In `compact_to_turn_bundle()`, preserve observation-first behavior and add lane-shaped metadata:
  - `plan_state`
  - `evidence_refs`
  - `decision_deltas`
  - `experience_candidates`
- In async brief compaction, include those fields in the JSON schema when possible.
- Keep `compact_recent_messages_async_with_status()` and `compact_recent_messages_with_status()` as emergency transcript fallback.

Storage targets:
- Evidence goes to `ReasoningGraph.evidence_records` when it is a fact/observation.
- Decisions go to `ReasoningGraph.decision_records` when they have rationale/evidence refs.
- Lessons go through existing experience memory paths.
- Current step, failures, open questions, and next actions go to `WorkingMemory`.

Tests:
- Update `tests/test_compaction_progress.py`.
- Assert observation-first compaction records lane-shaped metadata.
- Assert generic episodic summary fallback is used only for emergency/failure paths.
- Assert compaction demotion logs include lane counts and fallback reason.

## Phase 7: Defaults and Cleanup

Rollout:
1. Land compiler and tests with all flags off.
2. Enable `rewoo_planner_frame_enabled` for ToolPlan only.
3. Attach ToolPlan observations to `ReasoningGraph` evidence.
4. Enable `rewoo_solver_frame_enabled` for ToolPlan only.
5. Add refiner frames behind `rewoo_refiner_frame_enabled`.
6. Add lane-aware compaction metadata.
7. Flip `rewoo_lane_frames_enabled` default on for ToolPlan after evals improve or remain neutral.
8. Demote generic episodic summaries from normal ToolPlan prompts.

Cleanup after rollout:
- Remove the solver path that appends freeform observation system messages if no longer needed.
- Keep `render_tool_plan_observations()` for logs, diagnostics, trajectory recording, and legacy readability.
- Keep `_attach_tool_plan_evidence()` only if subtask-ledger evidence still needs a compact human-readable string.
- Keep generic summaries for audit/fallback, not primary ToolPlan prompt context.

## Evaluation Plan

Use existing ToolPlan evals under `evals/tool_plan/` and add prompt-shape assertions.

Metrics to track:
- planner prompt estimated tokens
- solver prompt estimated tokens
- `tool_plan_total_tokens`
- `tool_plan_observation_tokens`
- repeated read/search count
- invalid plan repair count
- unsafe/wrong-path rejection count
- solver missing-evidence requests
- patch-before-evidence failures
- refine pass/revise/block count
- task completion rate

Add assertions to `scripts/tool_plan_eval.py` or a companion report:
- planner prompt excludes transcript/tool chatter
- solver prompt includes current evidence ids
- solver prompt excludes unrelated warm summaries
- average ToolPlan prompt tokens decrease or stay flat
- completion and safety metrics improve or remain neutral

## Success Criteria

- Planner token use stays bounded while plan quality improves or stays neutral.
- Planner sees prior failures and open questions without seeing raw logs.
- Solver sees current evidence, failures, duplicates, and contradictions without rereading raw tool output.
- Refiner critiques against structured evidence instead of a large freeform blob.
- Repeated reads/searches decrease.
- Patch-before-evidence mistakes decrease.
- Generic older-chat summaries become fallback/audit context, not the primary ToolPlan/ReWOO substrate.
