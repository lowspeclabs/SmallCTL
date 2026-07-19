from __future__ import annotations

from .tool_plan_schema import PARALLELIZABLE_TOOL_PLAN_TOOLS


TOOL_PLAN_TOOL_LIST = " | ".join(sorted(PARALLELIZABLE_TOOL_PLAN_TOOLS))


def build_tool_plan_planner_prompt(
    *,
    task: str,
    max_steps: int,
    max_parallel: int = 4,
    context_frame: str = "",
) -> str:
    frame_section = f"""

Use this compact context frame to choose reads, but do not quote it or solve the task.

{context_frame}
""" if context_frame else ""
    return f"""You are the ToolPlan planner.

Your job is NOT to solve the task yet.
Your job is to produce a bounded read-only evidence plan.

Return ONLY JSON:

{{
  "mode": "tool_plan",
  "objective": "...",
  "steps": [
    {{
      "id": "E1",
      "tool": "{TOOL_PLAN_TOOL_LIST}",
      "args": {{"path": "relative/path"}},
      "reason": "...",
      "depends_on": []
    }}
  ]
}}

Rules:
- Max {max_steps} steps.
- Use read-only tools only.
- Do not solve the task in this response.
- Do not use shell_exec, ssh_exec, file_write, file_patch, ast_patch, task_complete, ask_human, memory_update, log_note, or ansible.
- Prefer targeted reads/searches over broad exploration.
- For source-code questions, search exact identifiers and likely snake_case names from the task, then read the matched source files.
- Do not add language-specific include filters unless the task or context clearly identifies that language.
- Do not spend a step on README or other docs unless the task asks for documentation or source searches fail.

Required arguments (do not leave args empty; use the real path/host/query for the task):
- file_read: {{"path": "relative/path"}}
- read_log: {{"path": "relative/path", "lines": 50}}
- dir_list: {{"path": "relative/dir"}}
- grep: {{"path": "relative/dir", "pattern": "term"}}
- find_files: {{"path": "relative/dir", "pattern": "*.py"}}
- ssh_file_read: {{"target": "user@host", "path": "/remote/path"}}
- ssh_dir_list: {{"target": "user@host", "path": "/remote/dir"}}
- artifact_read: {{"artifact_id": "..."}}
- artifact_grep: {{"artifact_id": "A0001", "query": "term"}}
- web_search: {{"query": "..."}}
- web_fetch: {{"url": "https://..."}}
- git_status: {{"path": "."}}
- git_diff: {{"path": "."}}

Dependency rules:
- Omit depends_on or set it to [] when a step does not need output from another step.
- Add depends_on only when this step literally needs a path, id, URL, query term, or other value discovered by an earlier step.
- Do not make file reads depend on searches unless the search result determines the path to read.
- Do not make web_search depend on file_read unless the query comes from that file.
- Independent steps may run concurrently. Max parallel width is {max(1, int(max_parallel))}.
{frame_section}

User task:
{task}"""


def build_tool_plan_solver_system_suffix(observations_text: str, *, fresh_output_limit: int = 1200) -> str:
    context_label = "REWOO CONTEXT FRAME" if "REWOO " in observations_text else "TOOL PLAN OBSERVATIONS"
    return f"""You are the ToolPlan solver.

Use the user task, active context, safety guidance, and {context_label}.

Now decide the next action:
- call task_complete with an answer grounded in the evidence
- call task_fail if the evidence is insufficient or the task cannot be completed from the evidence
- do not call read, search, shell, patch, or other normal tools from this solver turn
- Keep fresh explanatory output under about {max(1, int(fresh_output_limit))} characters unless the user explicitly asks for more.

{observations_text}"""
