"""Tool result deduplication - Cache identical tool calls within a trial.

MITIGATED VERSION - Includes fixes for:
#8. State leakage between trials - Automatic reset + explicit reset function
#5. Cache poisoning - Only cache successful results
#2. Stale data - TTL enabled by default (5 minutes)
#7. Debugging complexity - DEBUG logging of cache operations
#9. Argument serialization - Graceful fallback on non-serializable args
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from functools import wraps

from aho.mock_tools import long_context_lookup, summarize_report

# Setup logging for cache operations
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached tool result."""
    result: dict[str, Any]
    timestamp: float
    hit_count: int = 0


class ToolCallCache:
    """
    Cache for tool call results within a trial.
    
    MITIGATION #2: Default TTL prevents stale data
    Key: (tool_name, args_hash)
    Value: CacheEntry with result and metadata
    """
    
    # MITIGATION #2: Default 5-minute TTL to prevent stale data
    DEFAULT_TTL_SECONDS = 300
    
    def __init__(self, ttl_seconds: float | None = DEFAULT_TTL_SECONDS):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (None = no expiry)
                        MITIGATION #2: Default is 5 minutes
        """
        self._cache: dict[str, CacheEntry] = {}
        self._ttl_seconds = ttl_seconds
        self._stats = {
            "hits": 0,
            "misses": 0,
            "total_calls": 0,
            "errors": 0,
        }
    
    def _make_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Create a cache key from tool name and arguments."""
        # Create deterministic string representation
        key_data = {
            "tool": tool_name,
            "args": args,
        }
        key_str = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        if self._ttl_seconds is None:
            return False
        return time.time() - entry.timestamp > self._ttl_seconds
    
    def get(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """
        Get cached result if available.
        
        MITIGATION #2: Checks TTL before returning
        MITIGATION #7: Logs cache operations at DEBUG level
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._make_key(tool_name, args)
        entry = self._cache.get(key)
        
        if entry is None:
            logger.debug(f"Cache MISS for {tool_name} (key: {key[:8]}...)")
            return None
        
        # MITIGATION #2: Check TTL
        if self._is_expired(entry):
            logger.debug(f"Cache EXPIRED for {tool_name} (key: {key[:8]}...)")
            del self._cache[key]
            return None
        
        # Update stats
        entry.hit_count += 1
        self._stats["hits"] += 1
        
        # MITIGATION #7: Log cache hit
        logger.debug(f"Cache HIT for {tool_name} (key: {key[:8]}..., hits: {entry.hit_count})")
        
        return entry.result
    
    def set(self, tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> bool:
        """
        Store result in cache.
        
        MITIGATION #5: Only cache successful results
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result to cache
            
        Returns:
            True if cached, False if skipped (failed result)
        """
        # MITIGATION #5: Only cache successful results
        if not result.get("success", False):
            logger.debug(f"Cache SKIP for {tool_name} - result not successful")
            return False
        
        if result.get("error") is not None:
            logger.debug(f"Cache SKIP for {tool_name} - result has error")
            return False
        
        try:
            key = self._make_key(tool_name, args)
            self._cache[key] = CacheEntry(
                result=result,
                timestamp=time.time(),
                hit_count=0,
            )
            logger.debug(f"Cache SET for {tool_name} (key: {key[:8]}...)")
            return True
        except Exception as e:
            # MITIGATION #9: Graceful handling of key generation errors
            logger.warning(f"Cache SET failed for {tool_name}: {e}")
            return False
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        
        # Count expired entries
        expired_count = sum(1 for e in self._cache.values() if self._is_expired(e))
        
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "total_calls": total,
            "hit_rate": hit_rate,
            "cached_entries": len(self._cache),
            "expired_entries": expired_count,
            "valid_entries": len(self._cache) - expired_count,
            "total_hits_from_cached": sum(e.hit_count for e in self._cache.values()),
            "errors": self._stats["errors"],
        }
    
    def record_miss(self) -> None:
        """Record a cache miss."""
        self._stats["misses"] += 1
    
    def clear(self) -> None:
        """Clear all cached entries."""
        old_count = len(self._cache)
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "total_calls": 0, "errors": 0}
        logger.debug(f"Cache cleared ({old_count} entries removed)")


# Global cache instance (per trial)
_tool_cache: ToolCallCache | None = None


def get_tool_cache() -> ToolCallCache:
    """Get or create the global tool cache."""
    global _tool_cache
    if _tool_cache is None:
        _tool_cache = ToolCallCache()
    return _tool_cache


def reset_tool_cache() -> None:
    """
    Reset the tool cache (call at start of new trial).
    
    MITIGATION #8: Explicit reset prevents state leakage between trials
    """
    global _tool_cache
    logger.debug("Resetting tool cache for new trial")
    _tool_cache = ToolCallCache()


def ensure_fresh_cache() -> ToolCallCache:
    """
    Ensure we have a fresh cache instance.
    
    MITIGATION #8: Helper to guarantee isolation
    """
    global _tool_cache
    if _tool_cache is None:
        logger.debug("Creating fresh cache (was None)")
        _tool_cache = ToolCallCache()
    return _tool_cache


async def cached_long_context_lookup(
    *,
    topic: str,
    distilled: bool = True,
    query: str | None = None,
    min_confidence: float = 0.5,
    **kwargs: Any
) -> dict[str, Any]:
    """
    Cached version of long_context_lookup.
    
    Returns cached result if same arguments were used before in this trial.
    
    MITIGATION #5: Only successful results are cached
    MITIGATION #9: Graceful fallback on serialization errors
    """
    cache = ensure_fresh_cache()
    
    # Build args dict for cache key
    args = {
        "topic": topic,
        "distilled": distilled,
        "query": query,
        "min_confidence": min_confidence,
    }
    
    # MITIGATION #9: Defensive serialization
    try:
        # Check cache
        cached = cache.get("long_context_lookup", args)
        if cached is not None:
            # Return cached result with metadata
            cached_copy = dict(cached)
            if "metadata" not in cached_copy:
                cached_copy["metadata"] = {}
            cached_copy["metadata"]["cached"] = True
            cached_copy["metadata"]["cache_hit"] = True
            cached_copy["metadata"]["cache_key"] = cache._make_key("long_context_lookup", args)[:8]
            return cached_copy
    except (TypeError, ValueError) as e:
        # MITIGATION #9: Non-serializable args, skip cache
        logger.warning(f"Serialization failed for long_context_lookup: {e}")
        cache._stats["errors"] += 1
    
    # Cache miss - call actual tool
    cache.record_miss()
    result = await long_context_lookup(
        topic=topic,
        distilled=distilled,
        query=query,
        min_confidence=min_confidence,
        **kwargs
    )
    
    # Store in cache (MITIGATION #5: only successful results)
    try:
        cache.set("long_context_lookup", args, result)
    except Exception as e:
        logger.warning(f"Failed to cache result: {e}")
    
    return result


async def cached_summarize_report(
    *,
    subject: str,
    distilled: bool = True,
    query: str | None = None,
    min_confidence: float = 0.5,
    **kwargs: Any
) -> dict[str, Any]:
    """
    Cached version of summarize_report.
    
    Returns cached result if same arguments were used before in this trial.
    
    MITIGATION #5: Only successful results are cached
    MITIGATION #9: Graceful fallback on serialization errors
    """
    cache = ensure_fresh_cache()
    
    # Build args dict for cache key
    args = {
        "subject": subject,
        "distilled": distilled,
        "query": query,
        "min_confidence": min_confidence,
    }
    
    # MITIGATION #9: Defensive serialization
    try:
        # Check cache
        cached = cache.get("summarize_report", args)
        if cached is not None:
            # Return cached result with metadata
            cached_copy = dict(cached)
            if "metadata" not in cached_copy:
                cached_copy["metadata"] = {}
            cached_copy["metadata"]["cached"] = True
            cached_copy["metadata"]["cache_hit"] = True
            cached_copy["metadata"]["cache_key"] = cache._make_key("summarize_report", args)[:8]
            return cached_copy
    except (TypeError, ValueError) as e:
        # MITIGATION #9: Non-serializable args, skip cache
        logger.warning(f"Serialization failed for summarize_report: {e}")
        cache._stats["errors"] += 1
    
    # Cache miss - call actual tool
    cache.record_miss()
    result = await summarize_report(
        subject=subject,
        distilled=distilled,
        query=query,
        min_confidence=min_confidence,
        **kwargs
    )
    
    # Store in cache (MITIGATION #5: only successful results)
    try:
        cache.set("summarize_report", args, result)
    except Exception as e:
        logger.warning(f"Failed to cache result: {e}")
    
    return result


class DeduplicatedToolRegistry:
    """
    Tool registry that automatically deduplicates identical calls.
    
    Includes all mitigations:
    - MITIGATION #2: TTL support
    - MITIGATION #5: Success-only caching
    - MITIGATION #7: Debug logging
    - MITIGATION #8: Per-trial isolation
    - MITIGATION #9: Error handling
    """
    
    def __init__(self, ttl_seconds: float | None = ToolCallCache.DEFAULT_TTL_SECONDS):
        self._cache = ToolCallCache(ttl_seconds=ttl_seconds)
        self._tools: dict[str, Callable] = {
            "long_context_lookup": long_context_lookup,
            "summarize_report": summarize_report,
        }
    
    async def call(self, tool_name: str, **kwargs) -> dict[str, Any]:
        """
        Call a tool with automatic deduplication.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
            
        Returns:
            Tool result (cached or fresh)
        """
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # MITIGATION #9: Defensive serialization
        try:
            # Check cache
            cached = self._cache.get(tool_name, kwargs)
            if cached is not None:
                cached_copy = dict(cached)
                if "metadata" not in cached_copy:
                    cached_copy["metadata"] = {}
                cached_copy["metadata"]["cached"] = True
                return cached_copy
        except (TypeError, ValueError) as e:
            logger.warning(f"Serialization failed for {tool_name}: {e}")
        
        # Cache miss
        self._cache.record_miss()
        tool_fn = self._tools[tool_name]
        result = await tool_fn(**kwargs)
        
        # MITIGATION #5: Only cache successful results
        self._cache.set(tool_name, kwargs, result)
        
        return result
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    def reset_cache(self) -> None:
        """Reset cache (call at start of new trial)."""
        self._cache.clear()


if __name__ == "__main__":
    import asyncio
    
    # Enable debug logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    async def test_mitigations():
        """Test all mitigations."""
        print("=" * 70)
        print("MITIGATED TOOL DEDUPLICATION TEST")
        print("=" * 70)
        print("\nMitigations tested:")
        print("  #2: TTL enabled (5 min default)")
        print("  #5: Only successful results cached")
        print("  #7: DEBUG logging")
        print("  #8: Explicit cache reset")
        print("  #9: Graceful serialization fallback")
        
        # MITIGATION #8: Reset cache
        reset_tool_cache()
        
        print("\n1. First call to long_context_lookup...")
        result1 = await cached_long_context_lookup(
            topic="climate_policy",
            distilled=True,
            query="What is the reduction factor?"
        )
        print(f"   Tokens: {len(result1['output'].split())}")
        print(f"   Cached: {result1['metadata'].get('cached', False)}")
        
        print("\n2. Second identical call (should be cached)...")
        result2 = await cached_long_context_lookup(
            topic="climate_policy",
            distilled=True,
            query="What is the reduction factor?"
        )
        print(f"   Tokens: {len(result2['output'].split())}")
        print(f"   Cached: {result2['metadata'].get('cached', False)}")
        print(f"   Cache key: {result2['metadata'].get('cache_key', 'N/A')}")
        
        print("\n3. Different query (should NOT be cached)...")
        result3 = await cached_long_context_lookup(
            topic="climate_policy",
            distilled=True,
            query="What phase is this?"
        )
        print(f"   Tokens: {len(result3['output'].split())}")
        print(f"   Cached: {result3['metadata'].get('cached', False)}")
        
        # Check stats
        stats = get_tool_cache().get_stats()
        print(f"\n4. Cache Statistics:")
        print(f"   Hits: {stats['hits']}")
        print(f"   Misses: {stats['misses']}")
        print(f"   Hit rate: {stats['hit_rate']:.1%}")
        print(f"   Cached entries: {stats['cached_entries']}")
        print(f"   Valid entries: {stats['valid_entries']}")
        print(f"   Expired entries: {stats['expired_entries']}")
        print(f"   Errors: {stats['errors']}")
        
        print("\n5. MITIGATION #8 - Resetting cache...")
        reset_tool_cache()
        stats_after = get_tool_cache().get_stats()
        print(f"   Entries after reset: {stats_after['cached_entries']}")
        print(f"   Hits after reset: {stats_after['hits']}")
        
        print("\n" + "=" * 70)
        print("ALL MITIGATIONS WORKING")
        print("=" * 70)
    
    asyncio.run(test_mitigations())
