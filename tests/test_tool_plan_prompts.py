from __future__ import annotations

from smallctl.graph.tool_plan_prompts import build_tool_plan_planner_prompt


def test_tool_plan_planner_prompt_teaches_dependency_rules() -> None:
    prompt = build_tool_plan_planner_prompt(task="find dispatch code", max_steps=4, max_parallel=3)

    assert '"depends_on": []' in prompt
    assert "Omit depends_on or set it to []" in prompt
    assert "literally needs a path, id, URL, query term" in prompt
    assert "Do not add language-specific include filters" in prompt
    assert "Independent steps may run concurrently. Max parallel width is 3." in prompt
    assert "ssh_file_read" in prompt
    assert "git_status" in prompt
    assert "git_diff" in prompt
    assert "read_log" in prompt
