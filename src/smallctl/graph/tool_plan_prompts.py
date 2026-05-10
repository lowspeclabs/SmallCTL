from __future__ import annotations


TOOL_PLAN_TOOL_LIST = "file_read | dir_list | grep | find_files | artifact_read | artifact_grep | web_search | web_fetch"


def build_tool_plan_planner_prompt(*, task: str, max_steps: int) -> str:
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
      "args": {{}},
      "reason": "..."
    }}
  ]
}}

Rules:
- Max {max_steps} steps.
- Use read-only tools only.
- Do not solve the task in this response.
- Do not use shell_exec, ssh_exec, file_write, file_patch, ast_patch, task_complete, ask_human, memory_update, log_note, or ansible.
- Prefer targeted reads/searches over broad exploration.

User task:
{task}"""


def build_tool_plan_solver_system_suffix(observations_text: str, *, fresh_output_limit: int = 1200) -> str:
    return f"""You are the ToolPlan solver.

Use the user task, active context, safety guidance, and TOOL PLAN OBSERVATIONS.

Now decide the next action:
- answer
- propose or perform a patch through normal tools
- call a normal tool
- request a missing read
- stop if evidence is insufficient
- Keep fresh explanatory output under about {max(1, int(fresh_output_limit))} characters unless the user explicitly asks for more.

{observations_text}"""
