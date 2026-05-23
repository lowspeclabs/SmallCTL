"""
aho/similarity_retriever.py
Dynamic few-shot using similarity-based retrieval from past successes.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib


class SimpleSimilarityRetriever:
    """
    Lightweight similarity retriever using TF-IDF-like approach.
    No external dependencies - works with basic Python.
    """
    
    def __init__(self, results_path: str = "aho/results.jsonl", min_score: float = 0.85):
        self.results_path = Path(results_path)
        self.min_score = min_score
        self.successful_examples: List[Dict] = []
        self._load_successes()
    
    def _load_successes(self):
        """Load high-scoring completions from results history."""
        if not self.results_path.exists():
            return
        
        try:
            with open(self.results_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        result = json.loads(line)
                        score = result.get("mean_harness_score", 0)
                        if score >= self.min_score:
                            # Extract task description from strategy
                            strategy = result.get("strategy", {})
                            addendum = strategy.get("system_prompt_addendum", "")
                            
                            # Store the example
                            self.successful_examples.append({
                                "strategy_id": result.get("strategy_id", "unknown"),
                                "score": score,
                                "task_type": self._extract_task_type(addendum),
                                "keywords": self._extract_keywords(addendum),
                                "summary": f"Strategy {result.get('strategy_id')} achieved {score:.2f} score"
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
    
    def _extract_task_type(self, text: str) -> str:
        """Extract task type from text."""
        if "calculator" in text.lower():
            return "calculator"
        elif "analyzer" in text.lower():
            return "analyzer"
        elif "validator" in text.lower():
            return "validator"
        elif "engine" in text.lower():
            return "engine"
        return "generic"
    
    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        # Filter out common words
        stopwords = {'with', 'from', 'your', 'task', 'call', 'read', 'write', 'then', 'that', 'this', 'will'}
        return set(words) - stopwords
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_similar_example(self, current_task: str, top_k: int = 1) -> Optional[str]:
        """Retrieve most similar past successful example."""
        if not self.successful_examples:
            return None
        
        # Calculate similarity scores
        scored_examples = []
        for ex in self.successful_examples:
            sim = self._calculate_similarity(current_task, ex.get("task_type", ""))
            scored_examples.append((sim, ex))
        
        # Sort by similarity
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # Return top match if similarity is reasonable
        if scored_examples and scored_examples[0][0] > 0.1:
            best = scored_examples[0][1]
            return f"""### Similar Successful Strategy (Score: {best['score']:.2f})
Task Type: {best['task_type']}
Approach: {best['strategy_id']}
Key Insight: {best['summary']}
"""
        
        return None


def get_example_by_similarity(task_description: str) -> Optional[str]:
    """Convenience function to get similar example for a task."""
    retriever = SimpleSimilarityRetriever()
    return retriever.get_similar_example(task_description)
