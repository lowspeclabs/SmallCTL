import json
import os
import sys
import uuid
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

# Add the parent directory to sys.path to import src.smallctl
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(_REPO_ROOT / "src"))

from smallctl.client import OpenAICompatClient

class Hypothesis(BaseModel):
    id: str
    target_file: str
    target_lever: str
    proposed_value: Any
    rationale: str
    expected_impact: Dict[str, str] = Field(
        description="Key-value pairs of metric and expected change, e.g., {'mean_steps': '-15%', 'token_spend': '+5%'}"
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: str

class HypothesisEngine:
    def __init__(self, endpoint: str, model: str, api_key: str):
        self.api_client = OpenAICompatClient(
            base_url=endpoint,
            model=model,
            api_key=api_key,
            chat_endpoint="/chat/completions"
        )

    async def generate_hypothesis(self, static_analysis: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Hypothesis:
        """
        Main entry point for generating the single highest-confidence improvement proposal using an LLM.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(static_analysis, historical_data)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        chunks = []
        async for chunk in self.api_client.stream_chat(messages=messages, tools=[]):
            chunks.append(chunk)

        stream_result = OpenAICompatClient.collect_stream(chunks)
        assistant_text = stream_result.assistant_text.strip()
        
        # Extract JSON from assistant text
        try:
            if "```json" in assistant_text:
                json_str = assistant_text.split("```json")[1].split("```")[0].strip()
            elif "```" in assistant_text:
                json_str = assistant_text.split("```")[1].strip()
            else:
                json_str = assistant_text
            
            data = json.loads(json_str)
            
            # Ensure required fields exist in LLM response
            return Hypothesis(
                id=uuid.uuid4().hex[:8],
                target_file=data["target_file"],
                target_lever=data["target_lever"],
                proposed_value=data["proposed_value"],
                rationale=data["rationale"],
                expected_impact=data["expected_impact"],
                confidence_score=float(data["confidence_score"]),
                created_at=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            print(f"Error parsing hypothesis from LLM: {e}")
            print(f"Assistant Response: {assistant_text}")
            raise

    async def generate_crash_hypothesis(self, crash_trace: str) -> Hypothesis:
        """
        Generate an improvement proposal directed at fixing a specific runtime crash or bug tracelog.
        """
        system_prompt = self._build_system_prompt()
        system_prompt += "\nSPECIAL INSTRUCTION: Target a python logic bug fix to address the provided stacktrace."
        user_prompt = f"Crash Stacktrace/Details:\n{crash_trace}\n\nPropose a code fix."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        chunks = []
        async for chunk in self.api_client.stream_chat(messages=messages, tools=[]):
            chunks.append(chunk)

        stream_result = OpenAICompatClient.collect_stream(chunks)
        assistant_text = stream_result.assistant_text.strip()
        
        try:
            if "```json" in assistant_text:
                json_str = assistant_text.split("```json")[1].split("```")[0].strip()
            elif "```" in assistant_text:
                json_str = assistant_text.split("```")[1].strip()
            else:
                json_str = assistant_text
            
            data = json.loads(json_str)
            return Hypothesis(
                id=uuid.uuid4().hex[:8],
                target_file=data["target_file"],
                target_lever=data["target_lever"],
                proposed_value=data["proposed_value"],
                rationale=data["rationale"],
                expected_impact=data["expected_impact"],
                confidence_score=float(data["confidence_score"]),
                created_at=datetime.now(timezone.utc).isoformat()
            )
        except Exception as e:
            print(f"Error parsing crash hypothesis from LLM: {e}")
            raise

    def _build_system_prompt(self) -> str:
        return """You are the AHO (Agentic Harness Optimizer) Hypothesis Engine. 
You identify the single highest-confidence improvement to make to the smallctl harness.

Your output must be a JSON object inside a markdown code block:
{
  "target_file": "path/to/file.py",
  "target_lever": "VAR_NAME",
  "proposed_value": "new_value",
  "rationale": "one sentence explaining why this works",
  "expected_impact": {
    "success_rate": "+X%",
    "mean_steps": "-Y%",
    "token_spend": "+Z%"
  },
  "confidence_score": 0.85
}

RULES:
1. Target ONLY the extracted levers provided in the structural model.
2. Propose values that are likely to survive the A/B Oracle (significant improvement).
3. Do not propose redundant changes that failed in the past.
4. Ensure target_file is exactly as listed in the structural model.
"""

    def _build_user_prompt(self, static_analysis: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> str:
        history_str = json.dumps(historical_data, indent=2) if historical_data else "No historical trials recorded yet."
        analysis_str = json.dumps(static_analysis, indent=2)
        
        return f"""
Structural Model of Harness Levers:
{analysis_str}

Historical Trials & Learning Store:
{history_str}

Identify the best optimization move.
"""

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
    
    endpoint = os.getenv("SMALLCTL_ENDPOINT")
    model = os.getenv("SMALLCTL_MODEL")
    api_key = os.getenv("SMALLCTL_API_KEY")
    
    if not all([endpoint, model, api_key]):
        print("Missing environment variables in .env")
        sys.exit(1)
        
    engine = HypothesisEngine(endpoint, model, api_key)
    
    # Load analysis from stdin or a file for testing
    dummy_analysis = {
      "levers": [
        {"file_path": "src/smallctl/tools/artifact.py", "name": "DEFAULT_MAX_LINES", "current_value": 120}
      ]
    }
    
    async def main():
        hyp = await engine.generate_hypothesis(dummy_analysis, [])
        print(hyp.model_dump_json(indent=2))
        
    asyncio.run(main())
