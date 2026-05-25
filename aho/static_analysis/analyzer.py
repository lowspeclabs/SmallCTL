import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel


_HARNESS_INTERNAL_KEYS = {
    "allow_interactive_shell_approval",
    "artifact_read_inline_token_limit",
    "artifact_start_index",
    "policy",
    "run_logger",
    "shell_approval_session_default",
    "strategy",
    "strategy_prompt",
    "tool_result_inline_token_limit",
}


class HarnessLever(BaseModel):
    file_path: str
    name: str
    current_value: Any
    lever_type: str  # e.g., "constant", "prompt_template", "retry_count", "threshold"
    line_number: int
    context: Optional[str] = None


class DeadWiredConfig(BaseModel):
    field_name: str
    defined_in: str
    line_number: int
    default_value: Any


class Famalever(BaseModel):
    file_path: str
    name: str
    current_value: Any
    lever_type: str
    line_number: int
    context: Optional[str] = None


class ConfigParityReport(BaseModel):
    dead_wired: List[DeadWiredConfig]
    wired_but_not_defined: List[str]


class StaticAnalysisResult(BaseModel):
    levers: List[HarnessLever]
    file_summaries: Dict[str, str]
    config_parity: ConfigParityReport
    fama_levers: List[Famalever]


class HarnessStaticAnalyzer:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        # Only analyze src/smallctl
        self.target_dir = root_dir / "src" / "smallctl"

    def analyze(self) -> StaticAnalysisResult:
        levers = []
        file_summaries = {}

        for py_file in self.target_dir.rglob("*.py"):
            try:
                rel_path = str(py_file.relative_to(self.root_dir))
                levers.extend(self._extract_levers(py_file))
                file_summaries[rel_path] = self._summarize_file(py_file)
            except Exception as e:
                # Log but continue
                print(f"Error analyzing {py_file}: {e}")

        config_parity = self._analyze_config_parity()
        fama_levers = self._extract_fama_levers()

        return StaticAnalysisResult(
            levers=levers,
            file_summaries=file_summaries,
            config_parity=config_parity,
            fama_levers=fama_levers,
        )

    # ------------------------------------------------------------------
    # Generic lever extraction
    # ------------------------------------------------------------------
    def _extract_levers(self, file_path: Path) -> List[HarnessLever]:
        levers = []
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        rel_path = str(file_path.relative_to(self.root_dir))

        for node in ast.walk(tree):
            # 1. Look for constants at module level or inside classes
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        # Simple named constant
                        val = self._get_value_from_ast(node.value)
                        if val is not None:
                            levers.append(
                                HarnessLever(
                                    file_path=rel_path,
                                    name=target.id,
                                    current_value=val,
                                    lever_type="constant",
                                    line_number=node.lineno,
                                )
                            )

            # 2. Look for specifically named variables like "*_LIMIT", "*_TIMEOUT", "*_COUNT"
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and (
                    node.target.id.isupper()
                    or any(
                        s in node.target.id.upper()
                        for s in ["LIMIT", "TIMEOUT", "COUNT", "RATIO", "THRESHOLD"]
                    )
                ):
                    val = self._get_value_from_ast(node.value) if node.value else None
                    if val is not None:
                        levers.append(
                            HarnessLever(
                                file_path=rel_path,
                                name=node.target.id,
                                current_value=val,
                                lever_type="config_variable",
                                line_number=node.lineno,
                            )
                        )

        return levers

    def _get_value_from_ast(self, node: ast.AST) -> Any:
        try:
            val = ast.literal_eval(node)
            if isinstance(val, set):
                return list(val)
            return val
        except (ValueError, TypeError):
            # For complex values (like expressions or references), literal_eval fails
            return None

    def _summarize_file(self, file_path: Path) -> str:
        # Basic summary: first docstring or first 3 lines
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            if docstring:
                return docstring.split("\n")[0]

            lines = content.split("\n")
            non_empty = [l for l in lines if l.strip()][:2]
            return " | ".join(non_empty)

    # ------------------------------------------------------------------
    # Config parity analysis
    # ------------------------------------------------------------------
    def _analyze_config_parity(self) -> ConfigParityReport:
        config_fields = self._extract_smallctl_config_fields()
        harness_keys = self._extract_harness_init_keys()

        dead_wired: List[DeadWiredConfig] = []
        wired_but_not_defined: List[str] = []

        # Fields defined in SmallctlConfig but NOT wired into initialize_harness
        skipped = {
            "tui",
            "cleanup",
            "task",
            "config_path",
            "preset",
            "compatibility_warnings",
            "debug",
            "log_file",
            "api_key",
            "restore_graph_state",
            "graph_thread_id",
            "staged_reasoning",
        }
        for field_name, info in config_fields.items():
            if field_name in skipped:
                continue
            if field_name not in harness_keys:
                dead_wired.append(
                    DeadWiredConfig(
                        field_name=field_name,
                        defined_in=info["file"],
                        line_number=info["line"],
                        default_value=info["default"],
                    )
                )

        # Keys accepted by initialize_harness but NOT in SmallctlConfig
        for key in harness_keys:
            if key in _HARNESS_INTERNAL_KEYS:
                continue
            if key not in config_fields:
                wired_but_not_defined.append(key)

        return ConfigParityReport(
            dead_wired=dead_wired,
            wired_but_not_defined=wired_but_not_defined,
        )

    def _extract_smallctl_config_fields(self) -> Dict[str, Dict[str, Any]]:
        """Extract dataclass fields from src/smallctl/config.py"""
        config_path = self.target_dir / "config.py"
        fields: Dict[str, Dict[str, Any]] = {}
        if not config_path.exists():
            return fields

        with open(config_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SmallctlConfig":
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        if isinstance(item.target, ast.Name):
                            default_val = self._get_value_from_ast(item.value) if item.value else None
                            fields[item.target.id] = {
                                "file": str(config_path.relative_to(self.root_dir)),
                                "line": item.lineno,
                                "default": default_val,
                            }
        return fields

    def _extract_harness_init_keys(self) -> Set[str]:
        """Extract parameter names from initialize_harness in initialization.py"""
        init_path = self.target_dir / "harness" / "initialization.py"
        keys: Set[str] = set()
        if not init_path.exists():
            return keys

        with open(init_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        # Find the initialize_harness function and collect params.get("...", ...) or direct assignments
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "initialize_harness":
                for stmt in ast.walk(node):
                    # params["key"] or params.get("key", ...)
                    if isinstance(stmt, ast.Subscript):
                        if isinstance(stmt.value, ast.Name) and stmt.value.id == "params":
                            if isinstance(stmt.slice, ast.Constant) and isinstance(stmt.slice.value, str):
                                keys.add(stmt.slice.value)
                    elif isinstance(stmt, ast.Call):
                        if isinstance(stmt.func, ast.Attribute) and isinstance(stmt.func.value, ast.Name):
                            if stmt.func.value.id == "params" and stmt.func.attr == "get":
                                if stmt.args and isinstance(stmt.args[0], ast.Constant) and isinstance(stmt.args[0].value, str):
                                    keys.add(stmt.args[0].value)
        return keys

    # ------------------------------------------------------------------
    # FAMA policy extraction
    # ------------------------------------------------------------------
    def _extract_fama_levers(self) -> List[Famalever]:
        """Walk fama/tool_policy.py and fama/config.py for active mitigations and thresholds."""
        fama_dir = self.target_dir / "fama"
        levers: List[Famalever] = []

        # 1. Extract from fama/config.py
        config_file = fama_dir / "config.py"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for default values in function bodies / returns
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Constant) and isinstance(stmt.value, (int, float, str, bool)):
                            # Heuristic: named constants inside config accessors
                            if stmt.value in (2, 8, 3, 180, True, False):
                                levers.append(
                                    Famalever(
                                        file_path=str(config_file.relative_to(self.root_dir)),
                                        name=f"{node.name}_default",
                                        current_value=stmt.value,
                                        lever_type="fama_threshold",
                                        line_number=stmt.lineno,
                                        context=f"Default used in {node.name}",
                                    )
                                )

        # 2. Extract from fama/tool_policy.py
        policy_file = fama_dir / "tool_policy.py"
        if policy_file.exists():
            with open(policy_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            rel_path = str(policy_file.relative_to(self.root_dir))

            for node in ast.walk(tree):
                # Module-level sets (tool groups)
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            val = self._get_value_from_ast(node.value)
                            if val is not None:
                                levers.append(
                                    Famalever(
                                        file_path=rel_path,
                                        name=target.id,
                                        current_value=val,
                                        lever_type="fama_tool_set",
                                        line_number=node.lineno,
                                    )
                                )

                # Threshold comparisons (e.g., no_progress >= 2)
                if isinstance(node, ast.Compare):
                    for comp_node in ast.walk(node):
                        if isinstance(comp_node, ast.Constant) and isinstance(comp_node.value, (int, float)):
                            # Find the nearest statement line
                            line = getattr(comp_node, "lineno", 0)
                            levers.append(
                                Famalever(
                                    file_path=rel_path,
                                    name=f"threshold_at_line_{line}",
                                    current_value=comp_node.value,
                                    lever_type="fama_threshold",
                                    line_number=line,
                                    context="Comparison threshold in tool_policy",
                                )
                            )

                # Function-level defaults / config lookups
                if isinstance(node, ast.FunctionDef):
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Constant) and isinstance(stmt.value, (int, float, str, bool)):
                            if stmt.value in (2, 3, 0.9, True, False, "lite"):
                                levers.append(
                                    Famalever(
                                        file_path=rel_path,
                                        name=f"{node.name}_inline",
                                        current_value=stmt.value,
                                        lever_type="fama_threshold",
                                        line_number=stmt.lineno,
                                        context=f"Inline constant in {node.name}",
                                    )
                                )

        # Deduplicate by (file_path, name, line_number, current_value)
        seen: Set[tuple] = set()
        deduped: List[Famalever] = []
        for lev in levers:
            key = (lev.file_path, lev.name, lev.line_number, str(lev.current_value))
            if key not in seen:
                seen.add(key)
                deduped.append(lev)

        return deduped


if __name__ == "__main__":
    import json

    analyzer = HarnessStaticAnalyzer(Path("/home/stephen/Scripts/Harness-Redo"))
    result = analyzer.analyze()
    print(result.model_dump_json(indent=2))
