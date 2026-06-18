from __future__ import annotations

_RESPONSE_STRUCTURE_THINK = (
    "RESPONSE STRUCTURE: You MUST start EVERY response with a <think> block for plan and rationale. "
)

_RESPONSE_STRUCTURE_GEMMA = (
    "RESPONSE STRUCTURE: Start EVERY response with a <think> block for plan and rationale. "
    "Then continue with tool calls or the final answer as needed. "
)

_RESPONSE_STRUCTURE_SMALL_GEMMA = (
    "SMALL GEMMA-4 FORMAT: If you include short reasoning before a tool call, end the reasoning cleanly, "
    "then emit exactly one JSON tool object on its own line with no wrapper tags. "
)

_TOOL_CALL_FORMAT_JSON = (
    "TOOL CALL FORMAT: If tools are available, call them using the JSON format: "
    '`{"name": "tool_name", "arguments": {"arg": "val"}}`. '
)

_TOOL_CALL_FORMAT_TERMINAL = (
    "TERMINAL TOOL FORMAT: When the task is complete, emit the `task_complete` JSON tool call directly. "
    "Do not write `task_complete(message='...')` in chat text or Markdown. "
)

_TOOL_CALL_FORMAT_TERMINAL_SAME_TURN = (
    "TERMINAL TOOL FORMAT: If you have enough evidence to answer, call `task_complete` in the same turn "
    "that you formulate the answer. Do not send answer-only chat and wait for a later nudge. "
)

_WORKSPACE_RELATIVE_PATHS = (
    "WORKSPACE: Use relative paths (e.g. 'src/app.py'). You should prefer workspace-relative paths and do not start them with a leading slash or backslash. "
)

_PRIVILEGES_NO_SUDO_GUESS = (
    "PRIVILEGES: Do not invent or guess a sudo password. If privileged access is required, use passwordless sudo or ask the user for help via `ask_human`. "
)

_SHELL_POSIX_REDIRECTION = (
    "SHELL: Prefer standard POSIX redirection (e.g., `2>&1`) for robustness. "
)

_REMOTE_PROBES_BATCH = (
    "REMOTE PROBES: Batch read-only checks into single ssh_exec calls using && or ;. "
    "For commands longer than ~300 chars or with nested quotes, upload a temporary script "
    "(cat > /tmp/probe.sh << 'EOF' ...) and execute it. Use shlex.quote() for interpolated paths. "
)

_MEMORY_PERSIST_KEY_FACTS = "MEMORY: Use `memory_update` to persist key facts. "

_REDUNDANCY_PREFER_SUMMARY = (
    "REDUNDANCY: Prefer the compressed summary or preview first. Use `artifact_read` or `artifact_grep` only when you need the full evidence or line-level detail. "
)

_ARTIFACT_PAGING = (
    "ARTIFACT PAGING: When an artifact is truncated, page forward with `start_line` and `end_line` to get the next unseen chunk. Do not reread earlier chunks unless you need to verify a specific line. "
)

_REMOTE_DOWNLOAD_FALLBACK = (
    "DOWNLOAD: If curl fails repeatedly with 404 for a download URL, try wget before retrying the same URL. "
    "Different tools may resolve differently. If both curl and wget 404 on multiple URL variants, "
    "search the web for the correct source or ask the user for help."
)

_PATCH_VERBATIM_RULE = (
    "PATCH VERBATIM RULE: When using `file_patch` or `ast_patch`, copy the `target_text` verbatim from the most recent `file_read` or `artifact_print` output or artifact. "
    "Do not reconstruct target text from memory, summaries, or previews. If the file may have changed since your last read, re-read it immediately before patching. "
)

_EVIDENCE_ANCHORED_DIAGNOSIS_RULE = (
    "DIAGNOSIS RULE: Before stating a cause, anchor it to evidence with four fields: "
    "(1) observed line: the exact tool output line you are interpreting; "
    "(2) interpretation: what that line means in plain language; "
    "(3) confidence: high/medium/low and why; "
    "(4) next differentiating action: the one tool call that would confirm or refute the interpretation. "
    "Do not state causes like 'HTTP 503' or 'local DNS issue' unless a tool output explicitly supports them."
)

_SMALL_GEMMA_STRICT_FORMAT = (
    "SMALL GEMMA-4 STRICT FORMAT: Never emit `<tool_call>`, `<call>`, `<function=...>`, "
    "`<channel|>`, `<thought>`, angle-bracket function wrappers like `<task_complete(...)>`, "
    "or bare functional syntax like `dir_list()` or `task_complete(message='...')`. "
    "If tools are needed, emit only the JSON object. The backticked task_complete examples in this prompt "
    "describe intent only; do not copy that literal syntax into the response. "
)

_LFM_25_8B_STRICT_FORMAT = (
    "LFM2.5-8B STRICT FORMAT: Do not write JSON planning objects into chat. "
    "Never output top-level fields like `plan`, `next_actions`, `status_required`, or `next_step` as assistant text. "
    "If an action is needed, emit the actual tool call only. If no action is needed, answer briefly and call the terminal tool. "
    "Do not copy literal examples such as `task_complete(message=...)`; use the tool-call JSON protocol. "
)

_LARGE_GEMMA_26B_ANTI_LOOP_RULE = (
    "ANTI-LOOP RULE: Do not restart from the beginning of a task once concrete steps have already succeeded. "
    "If a directory was already created, a file was already written, or a container was already started, "
    "do not re-run the same creation/write/start command. Instead, verify the current state with a read or status command, "
    "then continue from the next missing step. Always trust the most recent tool result over older failure notes. "
    "If the current state is unclear, inspect it; do not blindly repeat setup steps. "
)

_LARGE_MODEL_STRUCTURED_REASONING = (
    "\n### STRUCTURED REASONING\n"
    "Use this framework for complex tasks:\n"
    "1. OBSERVE: What do I know from tools/memory?\n"
    "2. ORIENT: What's the gap between current state and goal?\n"
    "3. DECIDE: What's the single best next action?\n"
    "4. ACT: Execute the tool call.\n"
    "5. VERIFY: Did the result match expectations? If not, adjust.\n"
    "\n### SELF-CORRECTION\n"
    "Before calling task_complete, verify:\n"
    "- Did I fully address all parts of the task?\n"
    "- Are my conclusions supported by tool evidence (not inference)?\n"
    "- Did I skip any acceptance criteria?\n"
    "- If the task involves files, did I verify the final state?\n"
    "\n### PARALLEL EXECUTION\n"
    "When multiple independent facts are needed, make multiple tool calls in a single turn. "
    "For example, read 2-3 config files simultaneously rather than sequentially.\n"
    "\n### CONFIDENCE CALIBRATION\n"
    "If uncertain about a conclusion, say so. Prefer 'Based on the evidence, X seems likely but I cannot confirm Y' "
    "over definitive statements without evidence. When uncertain, gather more evidence rather than guessing.\n"
    "\n### FEW-SHOT EXAMPLES\n"
    "Good tool call: `{'name': 'file_read', 'arguments': {'path': 'src/app.py'}}`\n"
    "Bad (never do): `<tool_call>file_read(path='src/app.py')</tool_call>`\n"
    "Bad (never do): Writing `task_complete(message='...')` in chat text instead of proper JSON."
)

_CONTRACT_PHASE_FOCUS_SMALL: dict[str, str] = {
    "explore": "EXPLORE: Collect facts with read-only tools.",
    "plan": "PLAN: Draft an executable plan.",
    "author": "AUTHOR: Implement one bounded change at a time.",
    "execute": "EXECUTE: Run approved actions and verify.",
    "verify": "VERIFY: Compare results against acceptance criteria.",
}

_CONTRACT_PHASE_FOCUS_LARGE: dict[str, str] = {
    "explore": (
        "EXPLORE FOCUS: Collect verified observations, reduce uncertainty, and surface open questions before drafting a plan. "
        "Prefer read-only tools and concise fact capture."
    ),
    "plan": (
        "PLAN FOCUS: Rely on the compressed evidence packet, candidate causes, and handoff artifacts rather than a raw transcript dump. "
        "Turn observations into hypotheses and an executable plan."
    ),
    "author": (
        "AUTHOR FOCUS: Use the approved ExecutionPlan, the target files, and the active write session. "
        "Prefer one bounded implementation change at a time."
    ),
    "execute": (
        "EXECUTE FOCUS: Use the approved plan and evidence support, keep execution bounded to approved actions, and note the verification target."
    ),
    "verify": (
        "VERIFY FOCUS: Compare the observed state against the acceptance criteria and recent evidence. Prefer verification reads over new writes."
    ),
}

_REFLECTION_GATE = (
    "REFLECTION GATE: Before attempting the same fix again, you MUST explain in your reasoning: "
    "(1) what file the error message actually names, "
    "(2) what is currently in that file, and "
    "(3) why your previous fix did not affect that file. "
    "Do not call any repair tool until you have answered these three questions."
)

_META_COGNITIVE_REPAIR_BRIEF = (
    "META-COGNITIVE REPAIR BRIEF: If a configuration test or verifier fails, "
    "read the exact file path named in the error message before modifying any other file. "
    "Do not patch a source file (e.g., sites-available) when the error names a different path (e.g., sites-enabled)."
)

_STDERR_CIRCUIT_BREAKER_PREFIX = (
    "STDERR CIRCUIT BREAKER: The same stderr signature recurred twice: "
)

_LOCAL_ARTIFACT_TASK_PREFIX = (
    "LOCAL ARTIFACT TASK: Remote evidence collection is required, but the final report must be written to local path(s): "
)

_LOCAL_SCOPE_PREFERENCE = (
    "SCOPE: If the task mentions 'current user', 'this host', or local files such as ~/.ssh/known_hosts, "
    "prefer local file tools and shell_exec over ssh_exec, even if an IP address appears in the task text. "
    "The SSH client trust store (`~/.ssh/known_hosts`) is local to the harness machine; do not read or modify it with "
    "ssh_file_read, ssh_file_write, ssh_file_patch, or ssh_file_replace_between. "
)

_REMOTE_CLEANUP_TASK_KEYWORDS = (
    "uninstall", "remove", "delete", "purge", "clean up", "clean-up",
    "get rid of", "wipe", "tear down", "teardown", "disable",
)

_DELIVERABLE_VERIFICATION = (
    "DELIVERABLE VERIFICATION: Before calling `task_complete`, verify every file, path, or artifact explicitly requested in the task. "
    "If the task says to create a report at `/path/to/file`, use the appropriate read tool (e.g., `file_read` or `ssh_file_read`) to confirm the file exists and contains the expected content. "
    "Do not call `task_complete` based on the successful infrastructure step alone if a required deliverable is still missing. "
)

_DOCKER_INSPECT_HINT = (
    "DOCKER INSPECT HINT: To read container port mappings, use the correct Go template path: "
    "`docker inspect --format='{{json .NetworkSettings.Ports}}' <container>` or "
    "`docker inspect --format='{{json .HostConfig.PortBindings}}' <container>`. "
    "`.PortMappings` is not a valid key."
)

_INSTALLER_TIMEOUT_RECOVERY = (
    "INSTALLER TIMEOUT RECOVERY: If a remote installer or `docker run/pull` command times out but appears to be progressing, "
    "retry with a larger `timeout_sec` (e.g., 300) or run the command detached with output redirected to a log, then poll the log or service state. "
    "Do not abandon the task just because the first attempt exceeded the default timeout."
)

_PLANNING_MODE_INTRO = (
    "PLANNING MODE IS ACTIVE. "
    "Gather facts before proposing execution, use planning tools to create and refine a structured plan, "
    "and convert the plan into a playbook artifact that stages implementation into file skeleton, functions, code, and debug steps. "
    "Do not begin normal execution until the user explicitly approves the plan. "
    "Planning mode cannot execute shell commands. If phase validation requires running a verifier or test, call "
    "`request_validation_execution` with the exact command; after approval the loop runtime will use `shell_exec`. "
    "Never invent or call a tool named `run`. "
    "If you produce a draft plan, the harness will pause for approval automatically, so do not keep looping on the plan in the same turn. "
    "Do not call `task_complete` to exit planning; use `plan_request_execution` to pause for approval. "
    "Use plan export paths only for plan documents (.md, .txt, .text), never for implementation files like .py. "
    "Exactly one level of subplanning is allowed. "
    "The required plan shape is: goal, inputs, outputs, constraints, acceptance_criteria, implementation_plan, and steps. "
    "STEP TITLE RULE: Each step title must be a simple ≤6 word task name, imperative mood, and free of file paths. "
    "The title is ONLY a concise checklist label; put ALL details in `description` or `task`. "
    "Example good title: 'Write backoff script'. "
    "Example bad title: 'Build a self-contained Python script at ./temp/restart_backoff.py'."
)
