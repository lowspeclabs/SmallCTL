"""Dynamic fact extraction from tool results for inline display."""

from __future__ import annotations

import re
from typing import Any


class QueryContext:
    """Represents the context of a user query for targeted fact extraction."""
    
    def __init__(self, query: str | None = None, topic: str | None = None):
        self.query = query or ""
        self.topic = topic or self._detect_topic(query or "")
        self.intent = self._detect_intent(self.query)
        self.keywords = self._extract_keywords(self.query)
    
    def _detect_topic(self, query: str) -> str:
        """Auto-detect topic from query string."""
        query_lower = query.lower()
        
        topic_keywords = {
            "climate": ["climate", "reduction", "emissions", "ets", "phase iv", "2030"],
            "carbon": ["carbon", "eua", "price", "market", "allowance", "tonne"],
            "policy": ["policy", "regulation", "directive", "framework", "compliance"],
            "energy": ["energy", "renewable", "solar", "wind", "fossil", "fuel"],
            "trading": ["trading", "futures", "options", "settlement", "exchange"],
        }
        
        scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[topic] = score
        
        return max(scores, key=scores.get) if scores else "general"
    
    def _detect_intent(self, query: str) -> str:
        """Detect user intent from query."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["what is", "what's", "find", "get", "lookup"]):
            return "retrieval"
        elif any(w in query_lower for w in ["compare", "difference", "versus", "vs"]):
            return "comparison"
        elif any(w in query_lower for w in ["calculate", "compute", "determine"]):
            return "calculation"
        elif any(w in query_lower for w in ["summarize", "summary", "brief"]):
            return "summarization"
        
        return "general"
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        # Filter out stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "shall",
                      "can", "need", "dare", "ought", "used", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "as", "into",
                      "through", "during", "before", "after", "above", "below",
                      "between", "under", "and", "but", "or", "yet", "so"}
        return [w for w in words if w not in stop_words and len(w) > 2]


class FactExtractor:
    """
    Extracts relevant facts from documents based on query context.
    
    Phase 2: Smart Fact Extraction
    - Query-specific extraction (only facts relevant to the query)
    - Context-aware (understands what user is looking for)
    - Multi-domain pattern matching
    """
    
    # Enhanced pattern definitions for different domains
    FACT_PATTERNS = {
        "climate": [
            # Reduction factors (handles "from X% to Y%" format in document)
            (r"reduction\s+factor.*?from\s+\d+\.?\d*\s*%\s+to\s+(\d+\.?\d*)\s*%", "reduction_factor"),
            (r"(?:linear\s+)?reduction\s+factor.*?[:\s]+(\d+\.?\d*)%", "reduction_factor"),
            (r"(\d+\.?\d*)%\s*(?:annual\s+)?reduction", "reduction_factor"),
            (r"reduce.*?by\s*(\d+\.?\d*)%", "reduction_target"),
            
            # Phase information
            (r"EU\s+ETS\s+Phase\s+(IV|4|V|5|VI|6)", "phase"),
            (r"Phase\s+(IV|4|V|5)\s*\(\s*(\d{4})\s*[-–]\s*(\d{4})\s*\)", "phase_period"),
            
            # Targets and years
            (r"(\d+)%\s*.*?by\s*2030", "2030_target"),
            (r"(\d+)%\s*.*?by\s*2050", "2050_target"),
            (r"(\d{4})\s*[-–]\s*(\d{4})", "year_range"),
            
            # Emissions
            (r"(?:net\s+)?greenhouse[\s-]gas\s+emissions", "emissions_type"),
            (r"reduce.*?emissions.*?by\s*(\d+)%", "emissions_reduction"),
            (r"(\d+)%\s+reduction\s+vs\.?\s*(\d{4})", "reduction_baseline"),
        ],
        "carbon": [
            # Prices
            (r"EUA\s+Dec-\d+\s+futures\s+settled\s+at\s+€\s*(\d+\.?\d*)", "eua_price"),  # Most specific first
            (r"(?:EUA|carbon)\s+(?:Dec-\d+|allowances?)\s+(?:settled\s+)?at\s+€?\s*(\d+\.?\d*)", "eua_price"),
            (r"€\s*(\d+\.?\d*)[/\s]*t(?:onne)?", "carbon_price_per_tonne"),
            (r"(\d+\.?\d*)\s*€/t", "carbon_price_euros"),
            (r"(?:price|settled)\s+at.*?([€$£]\s*\d+\.?\d*)", "settlement_price"),
            (r"averaged\s+€?\s*(\d+\.?\d*)", "average_price"),
            
            # Changes
            (r"(\d+\.?\d*)%\s*(?:decrease|decline|drop|fall|down)", "price_decrease"),
            (r"(\d+\.?\d*)%\s*(?:increase|rise|gain|up)", "price_increase"),
            (r"(\d+\.?\d*)%\s*(?:YoY|year[\s-]over[\s-]year)", "yoy_change"),
            
            # Market info
            (r"(ICE\s+Endex|EEX|NASDAQ)", "exchange"),
            (r"(Q[1-4])\s+(\d{4})", "quarter_year"),
        ],
        "policy": [
            (r"Directive\s+(\d{4}/\d{2}/EU)", "directive_number"),
            (r"Regulation\s+(EU)\s+No?\.?\s*(\d+/\d{4})", "regulation_number"),
            (r"Phase\s+(I{1,3}|IV|V|VI|1|2|3|4|5|6)", "phase_number"),
            (r"target.*?[:\s]+(\d+\.?\d*)%", "target_percentage"),
            (r"(\d{4})\s*[-–]\s*(\d{4})", "implementation_period"),
            (r"(?:entry\s+into\s+force|effective)\s+(\d{1,2}\s+\w+\s+\d{4})", "effective_date"),
        ],
        "energy": [
            (r"(\d+\.?\d*)%\s+renewable", "renewable_percentage"),
            (r"(\d+)\s*GW\s+(?:of\s+)?(?:solar|wind|capacity)", "capacity_gw"),
            (r"(?:solar|wind|hydro|nuclear|gas|coal)\s+energy", "energy_source"),
        ],
        "trading": [
            (r"(\d+\.?\d*)\s+ million\s+(?:allowances|units)", "trading_volume"),
            (r"(?:futures|options)\s+(?:for|settled)\s+(\w+\s+\d{4})", "contract_month"),
        ],
    }
    
    # Query-to-fact relevance mapping
    QUERY_FACT_MAP = {
        "reduction_factor": ["reduction_factor", "reduction_target", "phase", "year_range"],
        "eua_price": ["eua_price", "carbon_price_per_tonne", "settlement_price", "average_price", "exchange", "quarter_year"],
        "phase": ["phase", "phase_period", "phase_number", "year_range", "implementation_period"],
        "target": ["2030_target", "2050_target", "target_percentage", "reduction_target"],
        "price": ["eua_price", "carbon_price_per_tonne", "settlement_price", "exchange", "price_decrease", "price_increase", "yoy_change"],
    }
    
    def __init__(self):
        self.compiled_patterns: dict[str, list[tuple]] = {}
        for domain, patterns in self.FACT_PATTERNS.items():
            self.compiled_patterns[domain] = [
                (re.compile(pattern, re.IGNORECASE), label)
                for pattern, label in patterns
            ]
    
    def extract_smart(
        self,
        document: str,
        query_context: QueryContext | None = None,
        max_facts: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Smart fact extraction based on query context.
        
        Phase 2: Only extracts facts relevant to the specific query.
        
        Args:
            document: Full text document
            query_context: Context about what the user is looking for
            max_facts: Maximum number of facts to return
            
        Returns:
            List of relevant extracted facts
        """
        if query_context is None:
            query_context = QueryContext()
        
        # Extract all facts from relevant domains
        all_facts = self._extract_all(document, query_context.topic)
        
        if not all_facts:
            return []
        
        # Score facts by relevance to query
        scored_facts = [
            (fact, self._relevance_score(fact, query_context))
            for fact in all_facts
        ]
        
        # Sort by relevance score (descending)
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        
        # Return top facts
        return [fact for fact, score in scored_facts[:max_facts] if score > 0]
    
    def _extract_all(self, document: str, topic: str | None = None) -> list[dict[str, Any]]:
        """Extract all facts from document (internal method)."""
        facts = []
        
        # Determine which domains to search
        if topic and topic in self.compiled_patterns:
            domains = [topic]
        else:
            domains = list(self.compiled_patterns.keys())
        
        for domain in domains:
            patterns = self.compiled_patterns.get(domain, [])
            for pattern, label in patterns:
                matches = pattern.findall(document)
                for match in matches:
                    # Handle tuple matches (groups)
                    if isinstance(match, tuple):
                        value = " ".join(m.strip() for m in match if m.strip())
                    else:
                        value = match.strip()
                    
                    # Skip duplicates
                    if any(f["value"] == value and f["label"] == label for f in facts):
                        continue
                    
                    facts.append({
                        "label": label,
                        "value": value,
                        "domain": domain,
                        "context": self._get_context(document, value),
                    })
        
        return facts
    
    def _relevance_score(self, fact: dict[str, Any], query_context: QueryContext) -> float:
        """
        Calculate relevance score of a fact to the query context.
        
        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check if fact label matches query keywords
        fact_label_lower = fact["label"].lower()
        for keyword in query_context.keywords:
            if keyword in fact_label_lower:
                score += 0.3
        
        # Check query-to-fact mapping
        for query_kw, relevant_labels in self.QUERY_FACT_MAP.items():
            if query_kw in query_context.keywords:
                if fact["label"] in relevant_labels:
                    score += 0.4
        
        # Domain match bonus
        if fact["domain"] == query_context.topic:
            score += 0.2
        
        # Priority bonus
        priority = self._priority(fact["label"])
        score += (priority / 100) * 0.1
        
        return min(1.0, score)
    
    def extract(
        self,
        document: str,
        query_topic: str | None = None,
        query_string: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract facts (backward compatible interface).
        
        Args:
            document: Full text document
            query_topic: Topic hint (climate, carbon, etc.)
            query_string: Full query string for context
            
        Returns:
            List of extracted facts
        """
        context = QueryContext(query=query_string or "", topic=query_topic)
        return self.extract_smart(document, context, max_facts=10)
    
    def format_inline(self, facts: list[dict], max_facts: int = 5) -> str:
        """
        Format facts for inline display in tool results.
        
        Args:
            facts: List of extracted fact dictionaries
            max_facts: Maximum number of facts to include
            
        Returns:
            Formatted string for LLM consumption
        """
        if not facts:
            return "No key facts extracted from document."
        
        lines = ["KEY FACTS:"]
        
        for fact in facts[:max_facts]:
            label = fact["label"].replace("_", " ").title()
            value = fact["value"]
            lines.append(f"• {label}: {value}")
        
        if len(facts) > max_facts:
            lines.append(f"... and {len(facts) - max_facts} more facts available")
        
        return "\n".join(lines)
    
    def format_structured(self, facts: list[dict]) -> dict[str, Any]:
        """
        Format facts as structured data for programmatic use.
        
        Returns:
            Dictionary organized by domain and label
        """
        structured: dict[str, Any] = {}
        
        for fact in facts:
            domain = fact["domain"]
            label = fact["label"]
            
            if domain not in structured:
                structured[domain] = {}
            
            structured[domain][label] = {
                "value": fact["value"],
                "context": fact["context"],
            }
        
        return structured
    
    def _get_context(self, document: str, match_value: str, window: int = 50) -> str:
        """Extract surrounding context for a match."""
        idx = document.find(match_value)
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(document), idx + len(match_value) + window)
        
        context = document[start:end]
        # Clean up
        context = context.replace("\n", " ").strip()
        if start > 0:
            context = "..." + context
        if end < len(document):
            context = context + "..."
        
        return context
    
    def _priority(self, label: str) -> int:
        """Return priority score for sorting facts."""
        priorities = {
            "reduction_factor": 100,
            "eua_price": 100,
            "carbon_price_per_tonne": 95,
            "settlement_price": 90,
            "phase": 85,
            "phase_period": 85,
            "2030_target": 80,
            "2050_target": 75,
            "reduction_target": 75,
            "exchange": 70,
            "quarter_year": 60,
            "price_decrease": 55,
            "price_increase": 55,
        }
        return priorities.get(label, 50)


def extract_key_facts(
    document: str,
    topic: str | None = None,
    query: str | None = None,
    max_facts: int = 5,
) -> str:
    """
    Convenience function to extract and format key facts.
    
    Args:
        document: Full text to extract from
        topic: Optional topic hint (climate, carbon, policy)
        query: Optional full query string for context-aware extraction
        max_facts: Maximum facts to return
        
    Returns:
        Formatted inline fact string
    """
    extractor = FactExtractor()
    context = QueryContext(query=query or "", topic=topic)
    facts = extractor.extract_smart(document, context, max_facts)
    return extractor.format_inline(facts, max_facts)


if __name__ == "__main__":
    # Test the extractor with Phase 2 smart extraction
    test_doc = """
    Global Climate Policy Reference Document v4.2
    
    The EU Emissions Trading System (EU ETS) Phase IV (2021-2030) operates
    with an annual linear reduction factor of 2.2%. This ambitious target
    aims to reduce net greenhouse-gas emissions by at least 55% by 2030 
    relative to 1990 levels.
    
    The carbon markets showed EUA Dec-25 futures settled at €56.10/t
    on 31 December 2025 (ICE Endex), representing a 12% decrease year-over-year.
    Average EU ETS allowances across Q4 2025 were €58.40/t.
    """
    
    extractor = FactExtractor()
    
    print("=" * 60)
    print("Phase 2: Smart Fact Extraction Tests")
    print("=" * 60)
    
    # Test 1: Query for reduction factor
    print("\n1. Query: 'What is the reduction factor?'")
    context = QueryContext(query="What is the reduction factor?")
    facts = extractor.extract_smart(test_doc, context, max_facts=3)
    print(extractor.format_inline(facts))
    
    # Test 2: Query for carbon price
    print("\n2. Query: 'Find the EUA price'")
    context = QueryContext(query="Find the EUA price")
    facts = extractor.extract_smart(test_doc, context, max_facts=3)
    print(extractor.format_inline(facts))
    
    # Test 3: Query for phase information
    print("\n3. Query: 'What phase is this?'")
    context = QueryContext(query="What phase is this?")
    facts = extractor.extract_smart(test_doc, context, max_facts=3)
    print(extractor.format_inline(facts))
    
    # Test 4: Backward compatible (no context)
    print("\n4. Backward compatible (topic='climate'):")
    facts = extractor.extract(test_doc, query_topic="climate")
    print(extractor.format_inline(facts, max_facts=5))
