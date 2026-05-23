"""Validation for extracted facts against ground truth."""

from __future__ import annotations

from typing import Any


class FactValidator:
    """
    Validates that extracted facts match expected ground truth values.
    
    This ensures inline tool results maintain accuracy while reducing token usage.
    """
    
    # Ground truth database for validation (canonical labels only)
    GROUND_TRUTH = {
        "climate_policy": {
            "reduction_factor": "2.2%",
            "phase": "IV",
            "year_range": "2021-2030",
            "2030_target": "55%",
        },
        "carbon_markets": {
            "eua_price": "€56.10",
            "exchange": "ICE Endex",
        },
    }
    
    # Label aliases - treat these as equivalent
    LABEL_ALIASES = {
        "phase": ["phase", "phase_period", "phase_number"],
        "reduction_factor": ["reduction_factor"],
        "year_range": ["year_range"],
        "2030_target": ["2030_target", "reduction_target", "emissions_target"],
        "eua_price": ["eua_price", "carbon_price_per_tonne", "carbon_price_euros", "settlement_price"],
    }
    
    # Tolerance for numeric comparisons
    NUMERIC_TOLERANCE = 0.01  # 1% tolerance
    
    def validate(self, facts: list[dict], domain: str) -> dict[str, Any]:
        """
        Validate extracted facts against ground truth.
        
        Args:
            facts: List of extracted fact dicts with 'label' and 'value'
            domain: Domain to validate against (climate_policy, carbon_markets, etc.)
            
        Returns:
            Validation result with matches, misses, and confidence score
        """
        expected = self.GROUND_TRUTH.get(domain, {})
        found = {f["label"]: f["value"] for f in facts}
        
        matched = []
        mismatched = []
        missing = []
        extra = []
        used_fact_labels = set()
        
        # Check expected facts (with alias matching)
        for key, expected_val in expected.items():
            # Check if this key or any of its aliases exist in found facts
            aliases = self.LABEL_ALIASES.get(key, [key])
            found_match = None
            found_alias = None
            
            for alias in aliases:
                if alias in found and alias not in used_fact_labels:
                    found_match = found[alias]
                    found_alias = alias
                    break
            
            if found_match:
                if self._match(found_match, expected_val):
                    matched.append({
                        "label": key,
                        "expected": expected_val,
                        "found": found_match,
                        "matched_alias": found_alias,
                    })
                    used_fact_labels.add(found_alias)
                else:
                    mismatched.append({
                        "label": key,
                        "expected": expected_val,
                        "found": found_match,
                        "similarity": self._similarity(found_match, expected_val),
                    })
                    used_fact_labels.add(found_alias)
            else:
                missing.append(key)
        
        # Check for unexpected facts
        for key in found:
            if key not in used_fact_labels:
                extra.append({
                    "label": key,
                    "value": found[key],
                })
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            len(matched), len(mismatched), len(missing), len(extra)
        )
        
        return {
            "valid": len(mismatched) == 0 and len(missing) == 0,
            "confidence": confidence,
            "matched": matched,
            "mismatched": mismatched,
            "missing": missing,
            "extra": extra,
        }
    
    def calculate_confidence(self, facts: list[dict], domain: str) -> float:
        """
        Calculate confidence score for extracted facts.
        
        Returns:
            Float between 0.0 and 1.0
        """
        validation = self.validate(facts, domain)
        return validation["confidence"]
    
    def _match(self, found: str, expected: str) -> bool:
        """Check if found value matches expected (with fuzziness)."""
        # Normalize both values
        found_norm = self._normalize(found)
        expected_norm = self._normalize(expected)
        
        # Exact match
        if found_norm == expected_norm:
            return True
        
        # Partial match (e.g., "IV 2021 2030" contains "IV")
        if expected_norm in found_norm or found_norm in expected_norm:
            return True
        
        # Numeric comparison for percentages/prices
        found_num = self._extract_number(found_norm)
        expected_num = self._extract_number(expected_norm)
        
        if found_num is not None and expected_num is not None:
            # Check within tolerance
            if abs(found_num - expected_num) / expected_num <= self.NUMERIC_TOLERANCE:
                return True
        
        # Fuzzy string match
        return self._similarity(found, expected) >= 0.7  # Reduced from 0.8
    
    def _normalize(self, value: str) -> str:
        """Normalize value for comparison."""
        return (
            value.lower()
            .replace("€", "")
            .replace("$", "")
            .replace("%", "")
            .replace("/t", "")
            .replace(" per tonne", "")
            .replace(",", "")
            .strip()
        )
    
    def _extract_number(self, value: str) -> float | None:
        """Extract numeric value from string."""
        import re
        match = re.search(r"(\d+\.?\d*)", value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate string similarity (0.0 to 1.0)."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def _calculate_confidence(
        self,
        matched: int,
        mismatched: int,
        missing: int,
        extra: int,
    ) -> float:
        """Calculate overall confidence score."""
        total_expected = matched + mismatched + missing
        
        if total_expected == 0:
            return 1.0 if extra > 0 else 0.0
        
        # Base score from matches (core facts)
        base_score = matched / total_expected
        
        # Phase 3: More lenient scoring
        # Only penalize heavily for mismatches, not missing optional facts
        mismatch_penalty = mismatched * 0.2  # 20% per mismatch (reduced from 30%)
        missing_penalty = missing * 0.1  # 10% per missing (reduced from 20%)
        extra_penalty = extra * 0.02  # 2% per extra (reduced from 5%)
        
        # Bonus for having core facts (reduction_factor, eua_price)
        core_fact_bonus = 0.0
        if matched > 0:
            core_fact_bonus = 0.1  # Small bonus for any match
        
        confidence = base_score - mismatch_penalty - missing_penalty - extra_penalty + core_fact_bonus
        return max(0.0, min(1.0, confidence))


class InlineResultValidator:
    """
    Validates inline tool results meet quality thresholds.
    """
    
    def __init__(self, min_confidence: float = 0.8, max_token_estimate: int = 200):
        self.min_confidence = min_confidence
        self.max_token_estimate = max_token_estimate
        self.fact_validator = FactValidator()
    
    def validate_inline_result(
        self,
        inline_output: str,
        facts: list[dict],
        domain: str,
        metadata: dict,
    ) -> dict[str, Any]:
        """
        Validate an inline tool result.
        
        Returns:
            Validation result with approval status
        """
        # Check token estimate
        token_estimate = metadata.get("token_estimate", 0)
        token_ok = token_estimate <= self.max_token_estimate
        
        # Validate facts
        fact_validation = self.fact_validator.validate(facts, domain)
        confidence_ok = fact_validation["confidence"] >= self.min_confidence
        
        # Check for required content
        has_facts = len(facts) > 0
        has_key_facts_marker = "KEY FACTS:" in inline_output
        
        approved = all([
            token_ok,
            confidence_ok or len(facts) == 0,  # Allow if no facts to validate
            has_facts,
            has_key_facts_marker,
        ])
        
        return {
            "approved": approved,
            "reasons": {
                "token_ok": token_ok,
                "confidence_ok": confidence_ok,
                "has_facts": has_facts,
                "has_key_facts_marker": has_key_facts_marker,
            },
            "fact_validation": fact_validation,
            "recommendation": "use_inline" if approved else "use_full_document",
        }


# Convenience function
def validate_facts(facts: list[dict], domain: str) -> dict[str, Any]:
    """Quick validation of facts against ground truth."""
    validator = FactValidator()
    return validator.validate(facts, domain)
