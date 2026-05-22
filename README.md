# SmallCTL

An agentic harness for **small language models** that need help staying effective on real, multi-step tasks.

SmallCTL is designed for situations where a smaller model can be useful, but cannot be trusted to freely improvise across a long chain of tools, logs, files, and environment details. The harness adds **staging, evidence handling, memory compression, recovery, and risk controls** around the model so it can act more like a disciplined operator than an unbounded chatbot.

## What SmallCTL is for

SmallCTL is built to make smaller local or self-hosted models more useful for:

* **Coding-agent workflows**

  * repository exploration
  * targeted refactors and patching
  * verifier-driven edit loops
  * bounded file authoring and repair
* **Technical investigation and diagnosis**

  * log triage
  * evidence gathering
  * hypothesis building
  * verification against observed state
* **Structured tool use**

  * turning a raw model into a tool-using worker with state, persistence, and recovery
* **Longer-horizon task execution**

  * tasks that are too large or too messy for a single prompt, but still benefit from local-model economics

The harness is especially aimed at the failure modes common to smaller models:

* weak long-chain reasoning across many moving parts
* confusion when logs, tools, and environment details are mixed together
* incomplete-context hallucinations
* premature risky decisions
* brittle file writing on large outputs
* looping or repeating the same ineffective action

## Core design goals

SmallCTL exists to do five things well:

### 1\. Keep reasoning bounded and staged

Instead of letting the model freestyle from prompt to prompt, SmallCTL uses explicit phases to separate work such as:

* **explore** — gather facts and open questions
* **plan** — turn evidence into hypotheses and executable steps
* **author** — prepare code/content changes
* **execute** — run approved actions
* **verify** — compare observed state with expected results
* **repair** — recover from failed verification or execution

The goal is to stop the model from mixing discovery, speculation, mutation, and verification in one noisy loop.

### 2\. Make evidence first-class

SmallCTL is oriented around evidence-backed action instead of pure transcript reasoning. The state model supports structured evidence, decisions, claims, and context briefs so the harness can keep track of:

* what was observed
* what was inferred
* what remains unconfirmed
* what decisions were taken
* what evidence supported them

This is meant to reduce false certainty and give the harness a basis for safer verification and recovery.

### 3\. Compress state without losing task-critical context

Small models cannot carry an ever-growing raw transcript forever. SmallCTL uses layered context compaction and structured prompt assembly so older context can be demoted into:

* normalized observations
* turn bundles
* warm context briefs
* episodic summaries
* artifact snippets
* reusable experience memory

The goal is to preserve task continuity without forcing the model to reread the full raw history every turn.

### 4\. Treat tool execution as a controlled runtime, not chat decoration

Tool calls are not just text in the transcript. SmallCTL has an explicit execution and persistence path so tool work can be:

* dispatched in a controlled way
* recorded with metadata
* turned into artifacts
* reused by later steps
* reintroduced into prompts in a structured form

This makes the harness more like an agent runtime than a chat wrapper.

### 5\. Bound unsafe behavior

SmallCTL is designed for useful autonomy, not blind autonomy. The harness includes phase contracts, blocked-tool handling, risk policy surfaces, approval flows, and write-session recovery/loop control so smaller models are less likely to drift into unsafe or low-value behavior.

## Current feature themes

## Staged reasoning and execution

SmallCTL has explicit multi-phase task handling and staged reasoning support. The repo exposes phase contracts and a staged reasoning rollout path, making it possible to separate discovery, planning, authoring, execution, verification, and repair rather than forcing everything through a single ReAct-style loop.

## Structured state and reasoning records

The runtime keeps a rich `LoopState` and related records for things like:

* context briefs
* evidence records
* decision records
* claim records
* write sessions
* experience memory

This gives the harness a persistent working model of the task rather than relying only on raw recent messages.

## Prompt-state compilation and compaction

Prompt assembly is handled through a structured assembler that compiles a prompt frame from state, summaries, artifact snippets, and retrieved experience. The assembled prompt budgets space across lanes such as:

* recent messages
* run brief
* working memory
* FAMA capsules
* recovery guidance
* normalized observations
* fresh tool outputs
* turn bundles
* warm briefs
* episodic summaries
* artifact snippets
* warm memories

This is one of the main mechanisms that makes SmallCTL suitable for smaller-context local models.

## Evidence-oriented memory

SmallCTL supports context briefs that capture not just what happened, but also:

* confirmed facts
* unconfirmed facts
* open questions
* candidate causes
* disproven causes
* next observations needed

That pushes the harness toward evidence-led diagnosis instead of transcript-led guessing.

## Write-session control and recovery

Large file writing is treated as a controlled session, not a single uncontrolled generation. The write-session machinery supports modes such as:

* `single\_write`
* `chunked\_author`
* `local\_repair`
* `stub\_and\_fill`

The harness also includes a chunked-write loop guard and recovery behavior so repeated bad section writes can force recovery steps instead of endlessly repeating the same failure.

## Risk and approval surfaces

SmallCTL includes dedicated approval and risk-policy surfaces in the harness. The intended operating model is that the model can gather evidence, propose changes, and operate under policy, while the harness enforces when approval or stronger justification is required.

## Tool graph and execution plumbing

The repo includes a graph-oriented execution layer with components for:

* checkpoints
* interrupts
* progress tracking
* error hardening
* tool DAG support
* tool execution persistence
* tool execution recovery

This is the systems backbone that makes it possible to evolve beyond simple one-step tool calling.

## FAMA and reflexion-related support

The codebase includes a dedicated `fama` package with capsules, detectors, a router, a runtime, a reflexion bridge, signals, and tool-policy support. That suggests the harness is already experimenting with more adaptive or meta-reasoning style control rather than only static prompting.

## Memory, diagnostics, and operator-facing tooling

Beyond the main runtime, the repo includes memory CLI support, diagnostic-task helpers, cleanup/logging utilities, a search server package, and a Textual-based UI path. This points to SmallCTL being a practical operator harness, not just a library experiment.

## What SmallCTL is **not**

SmallCTL is not trying to pretend a 4B–9B class model is a fully reliable autonomous senior engineer.

It is not built around the assumption that the model should:

* freely make risky remediation decisions
* mutate systems without evidence or guardrails
* carry the whole task in raw chat memory
* be trusted purely because it can emit plausible text

Instead, SmallCTL is a harness for making smaller models **more usable, more inspectable, and less fragile**.

## Best-fit use cases

SmallCTL is a strong fit for:

* codebase exploration and patching
* verifier-driven coding loops
* sysadmin and ops triage with guarded execution
* repository and environment investigation
* long-running local-model experiments
* research into small-model tool use, memory, staged reasoning, and recovery

It is especially valuable when you want:

* lower-cost local inference
* tighter control over runtime behavior
* more explicit state and evidence than a normal chat interface provides
* a platform for trying reasoning/runtime upgrades without retraining a model

## Recommended operating philosophy

Use SmallCTL as:

* a **disciplined coding agent**
* a **diagnostic worker with evidence requirements**
* a **tool-using harness for weaker local models**
* a **research platform for staged reasoning, compaction, verification, and recovery**

Do **not** use it as a fully unbounded autonomous executor unless the surrounding risk policy, verifier surfaces, and approvals are strong enough for the environment.

## Near-term direction

Based on the current codebase and the design direction discussed in this project, the most valuable next upgrades are:

* tool-integrated candidate generation plus verification for hard coding/diagnostic steps
* compiler-lite parallelization of clearly independent read-only steps
* stronger verifier-first task completion rules
* even tighter state compression around evidence lanes and phase-specific retrieval
* clearer README and operator docs that describe the harness as it exists now, not only as a bundle installer
* a future web UI layered on top of the current harness/runtime seams

## In one sentence

**SmallCTL is a staged, evidence-aware, tool-driven harness that tries to make small language models useful on real multi-step technical work without pretending they are safe or reliable when left completely on their own.**

