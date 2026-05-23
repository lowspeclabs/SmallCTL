import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone

class KnowledgeStore:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._ensure_storage()

    def _ensure_storage(self):
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text("[]\n")

    def log_trial(self, hypothesis: Any, result: Dict[str, Any]):
        """
        Record the trial outcome into a knowledge store.
        """
        # We handle pydantic models or dicts
        data = {
            "hypothesis": hypothesis.model_dump() if hasattr(hypothesis, "model_dump") else hypothesis,
            "actual_outcome": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            current_data = json.loads(self.storage_path.read_text())
        except (json.JSONDecodeError, ValueError):
            current_data = []
            
        current_data.append(data)
        self.storage_path.write_text(json.dumps(current_data, indent=2))

    def get_past_outcomes(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.storage_path.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
