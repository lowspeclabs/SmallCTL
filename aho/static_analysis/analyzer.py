import ast
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class HarnessLever(BaseModel):
    file_path: str
    name: str
    current_value: Any
    lever_type: str  # e.g., "constant", "prompt_template", "retry_count", "threshold"
    line_number: int
    context: Optional[str] = None

class StaticAnalysisResult(BaseModel):
    levers: List[HarnessLever]
    file_summaries: Dict[str, str]

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

        return StaticAnalysisResult(levers=levers, file_summaries=file_summaries)

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
                            levers.append(HarnessLever(
                                file_path=rel_path,
                                name=target.id,
                                current_value=val,
                                lever_type="constant",
                                line_number=node.lineno
                            ))
            
            # 2. Look for specifically named variables like "*_LIMIT", "*_TIMEOUT", "*_COUNT"
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and (
                    node.target.id.isupper() or 
                    any(s in node.target.id.upper() for s in ["LIMIT", "TIMEOUT", "COUNT", "RATIO", "THRESHOLD"])
                ):
                    val = self._get_value_from_ast(node.value) if node.value else None
                    if val is not None:
                        levers.append(HarnessLever(
                            file_path=rel_path,
                            name=node.target.id,
                            current_value=val,
                            lever_type="config_variable",
                            line_number=node.lineno
                        ))

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

if __name__ == "__main__":
    import json
    analyzer = HarnessStaticAnalyzer(Path("/home/stephen/Scripts/Harness-Redo"))
    result = analyzer.analyze()
    print(result.model_dump_json(indent=2))
