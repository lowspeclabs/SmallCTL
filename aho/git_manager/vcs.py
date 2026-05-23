import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
from git import Repo

class GitManager:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.original_branch = self.repo.active_branch.name

    def is_dirty(self, untracked_files: bool = False) -> bool:
        return self.repo.is_dirty(untracked_files=untracked_files)

    def create_experiment_branch(self, hypothesis_id: str) -> str:
        branch_name = f"aho-experiment-{hypothesis_id}"
        # Ensure we're on a clean baseline (e.g. main)
        # self.repo.heads.main.checkout() 
        current = self.repo.create_head(branch_name)
        current.checkout()
        return branch_name

    def apply_hypothesis_patch(self, hypothesis: Dict[str, Any]) -> bool:
        """
        Attempts to apply the hypothesis change by replacing the old value of the lever.
        Very primitive implementation using simple string replace for demo.
        In production, this would use the line number from the static analysis result.
        """
        target_file = self.repo_path / hypothesis["target_file"]
        lever_name = hypothesis["target_lever"]
        new_val = hypothesis["proposed_value"]
        
        # Need original value to do a clean swap
        # The static analyzer should have provided this
        # For this demo, let's assume we search for 'LEVER_NAME = <ANY_VAL>'
        try:
            content = target_file.read_text(encoding="utf-8")
            import re
            
            # Pattern for: NAME = VALUE or NAME: TYPE = VALUE
            # \g<1> captures 'NAME: TYPE = ' or 'NAME = '
            pattern = rf"({lever_name}\s*(?::\s*[\w\[\]\"\'\s|]+)?\s*=\s*)[\w\"\'\d\.\-\[\]]+"
            replacement = rf"\g<1>{repr(new_val)}"
            
            new_content, count = re.subn(pattern, replacement, content)
            
            if count > 0:
                target_file.write_text(new_content, encoding="utf-8")
                return True
            else:
                # Fallback for just NAME: TYPE (no assignment) or simple consts
                # Search for 'NAME = ' followed by something
                pattern_simple = rf"({lever_name}\s*[:=]\s*)[\w\"\'\d\.\-\[\]]+"
                new_content, count = re.subn(pattern_simple, replacement, content)
                if count > 0:
                    target_file.write_text(new_content, encoding="utf-8")
                    return True
                
                print(f"Lever {lever_name} not found in {target_file}")
                return False
        except Exception as e:
            print(f"Error applying patch: {e}")
            return False

    def commit_hypothesis(self, hypothesis: Dict[str, Any]):
        """
        Commits with a full hypothesis JSON in the commit message.
        """
        self.repo.git.add(update=True)
        id = hypothesis["id"]
        msg = f"aho[hypothesis-{id}]: {hypothesis['rationale']}\n\nEXPERIMENT DETAILS:\n"
        msg += f"Target File: {hypothesis['target_file']}\n"
        msg += f"Target Lever: {hypothesis['target_lever']}\n"
        msg += f"Proposed Value: {hypothesis['proposed_value']}\n"
        msg += f"Expected Impact: {hypothesis['expected_impact']}\n"
        msg += f"Confidence: {hypothesis['confidence_score']}\n"
        msg += f"\nHYPOTHESIS_JSON: {json.dumps(hypothesis, indent=2)}"
        
        self.repo.index.commit(msg)

    def revert_to_original(self):
        self.repo.heads[self.original_branch].checkout()

    def merge_success(self, branch_name: str):
        self.repo.heads[self.original_branch].checkout()
        self.repo.git.merge(branch_name)
        # Maybe delete the experiment branch?
        # self.repo.git.branch("-d", branch_name)

if __name__ == "__main__":
    import json
    # Mocking for testing
    mgr = GitManager(Path("/home/stephen/Scripts/Harness-Redo"))
    # mgr.create_experiment_branch("test-id")
    # mgr.apply_hypothesis_patch({"target_file": "aho/pyproject.toml", "target_lever": "version", "proposed_value": "0.1.1"})
    # mgr.commit_hypothesis({"id": "test-id", "rationale": "testing git manager", "target_file": "...", "target_lever": "...", "proposed_value": "...", "expected_impact": {}, "confidence_score": 0.99})
