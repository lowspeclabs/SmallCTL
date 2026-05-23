"""Auto-batching tool dispatcher for parallel execution.

Automatically detects and batches independent tool calls for parallel execution.
Wraps the existing dispatcher to add auto-batching capability.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from collections import deque

from src.smallctl.tools.dispatcher import ToolInterceptor
from src.smallctl.models.tool_result import ToolEnvelope


@dataclass
class PendingToolCall:
    """A tool call waiting to be dispatched."""
    tool_name: str
    arguments: dict[str, Any]
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)


class AutoBatchDispatcher:
    """
    Auto-batching tool dispatcher.
    
    Collects tool calls over a short window and batches independent calls
    for parallel execution. Calls with dependencies (same tool or sequential
    requirements) are executed sequentially.
    
    Usage:
        # Wrap existing dispatcher
        original_dispatch = harness.dispatcher.dispatch
        batch_dispatcher = AutoBatchDispatcher(original_dispatch)
        harness.dispatcher.dispatch = batch_dispatcher.dispatch
        
        # Tool calls are now auto-batched
        result1 = await batch_dispatcher.dispatch("tool_a", {...})
        result2 = await batch_dispatcher.dispatch("tool_b", {...})
        # Executes in parallel if independent
    """
    
    def __init__(
        self,
        batch_window_ms: float = 50.0,  # Collect calls for 50ms
        max_batch_size: int = 5,
    ):
        self._batch_window_ms = batch_window_ms
        self._max_batch_size = max_batch_size
        self._next_dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolEnvelope]] | None = None
        
        # Pending calls queue
        self._pending: deque[PendingToolCall] = deque()
        self._batch_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        
        # Stats
        self._stats = {
            "total_calls": 0,
            "batched_calls": 0,
            "sequential_calls": 0,
            "batches_executed": 0,
        }
    
    async def __call__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        next_dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolEnvelope]],
    ) -> ToolEnvelope:
        """
        Intercept tool call for auto-batching.
        """
        self._next_dispatch = next_dispatch
        async with self._lock:
            self._stats["total_calls"] += 1
            
            # Create future for this call
            future = asyncio.get_event_loop().create_future()
            pending_call = PendingToolCall(tool_name, arguments, future)
            
            # Add to pending queue
            self._pending.append(pending_call)
            
            # Start batch timer if not already running
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(
                    self._execute_batch_after_delay()
                )
            
            # If queue is full, trigger batch immediately
            if len(self._pending) >= self._max_batch_size:
                if self._batch_task and not self._batch_task.done():
                    self._batch_task.cancel()
                self._batch_task = asyncio.create_task(self._execute_batch())
        
        # Wait for result
        try:
            return await future
        except asyncio.CancelledError:
            # Call was batched and will be resolved by batch execution
            return await future
    
    async def _execute_batch_after_delay(self) -> None:
        """Wait for batch window then execute batch."""
        try:
            await asyncio.sleep(self._batch_window_ms / 1000)
            await self._execute_batch()
        except asyncio.CancelledError:
            # Batch was executed early (queue full)
            pass
    
    async def _execute_batch(self) -> None:
        """Execute all pending calls as a batch."""
        async with self._lock:
            if not self._pending:
                return
            
            # Collect calls to execute
            calls_to_execute = list(self._pending)
            self._pending.clear()
        
        # Separate into batches of independent calls
        batches = self._group_into_batches(calls_to_execute)
        
        # Execute batches
        for batch in batches:
            if len(batch) == 1:
                # Single call - execute directly
                await self._execute_single(batch[0])
            else:
                # Multiple independent calls - execute in parallel
                await self._execute_parallel_batch(batch)
    
    def _group_into_batches(
        self, 
        calls: list[PendingToolCall]
    ) -> list[list[PendingToolCall]]:
        """
        Group calls into batches of independent operations.
        
        Calls are independent if:
        - They are different tools
        - They don't depend on each other's results
        
        Conservative approach: Only batch calls to DIFFERENT tools.
        """
        batches: list[list[PendingToolCall]] = []
        current_batch: list[PendingToolCall] = []
        current_tools: set[str] = set()
        
        for call in calls:
            if call.tool_name in current_tools:
                # Same tool - start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [call]
                current_tools = {call.tool_name}
            else:
                # Different tool - add to current batch
                current_batch.append(call)
                current_tools.add(call.tool_name)
                
                # Check batch size limit
                if len(current_batch) >= self._max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_tools = set()
        
        # Don't forget remaining calls
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _execute_single(self, call: PendingToolCall) -> None:
        """Execute a single tool call."""
        try:
            result = await self._next_dispatch(
                call.tool_name, 
                call.arguments
            )
            call.future.set_result(result)
            self._stats["sequential_calls"] += 1
        except Exception as e:
            call.future.set_exception(e)
    
    async def _execute_parallel_batch(
        self, 
        batch: list[PendingToolCall]
    ) -> None:
        """Execute a batch of independent tool calls in parallel."""
        self._stats["batches_executed"] += 1
        self._stats["batched_calls"] += len(batch)
        
        # Create tasks for all calls
        tasks = []
        for call in batch:
            task = asyncio.create_task(
                self._next_dispatch(call.tool_name, call.arguments)
            )
            tasks.append((call, task))
        
        # Wait for all to complete and resolve futures
        for call, task in tasks:
            try:
                result = await task
                call.future.set_result(result)
            except Exception as e:
                call.future.set_exception(e)
    
    def get_stats(self) -> dict[str, Any]:
        """Get dispatcher statistics."""
        total = self._stats["total_calls"]
        batched = self._stats["batched_calls"]
        return {
            **self._stats,
            "batch_rate": batched / total if total > 0 else 0,
            "avg_batch_size": batched / self._stats["batches_executed"] 
                if self._stats["batches_executed"] > 0 else 0,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_calls": 0,
            "batched_calls": 0,
            "sequential_calls": 0,
            "batches_executed": 0,
        }


class BatchingHarnessWrapper:
    """
    Wrapper that adds auto-batching to a harness instance.
    
    Usage:
        harness = Harness(...)
        wrapper = BatchingHarnessWrapper(harness)
        wrapper.enable_batching()
        
        # Now all tool calls are auto-batched
        result = await harness.run_task(task)
        
        stats = wrapper.get_batch_stats()
    """
    
    def __init__(self, harness: Any):
        self._harness = harness
        self._batch_dispatcher: AutoBatchDispatcher | None = None
        self._original_dispatch: Callable | None = None
    
    def enable_batching(
        self,
        batch_window_ms: float = 50.0,
        max_batch_size: int = 5,
    ) -> None:
        """Enable auto-batching on the wrapped harness."""
        if self._batch_dispatcher is not None:
            return  # Already enabled
        
        # Store original dispatch
        self._original_dispatch = self._harness.dispatcher.dispatch
        
        # Create batch dispatcher
        self._batch_dispatcher = AutoBatchDispatcher(
            original_dispatch=self._original_dispatch,
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
        )
        
        # Replace dispatch method
        self._harness.dispatcher.dispatch = self._batch_dispatcher.dispatch
    
    def disable_batching(self) -> None:
        """Disable auto-batching and restore original behavior."""
        if self._batch_dispatcher is None:
            return
        
        # Restore original dispatch
        self._harness.dispatcher.dispatch = self._original_dispatch
        self._batch_dispatcher = None
    
    def get_batch_stats(self) -> dict[str, Any]:
        """Get batching statistics."""
        if self._batch_dispatcher is None:
            return {"enabled": False}
        return {
            "enabled": True,
            **self._batch_dispatcher.get_stats(),
        }
    
    def __getattr__(self, name: str) -> Any:
        """Delegate to wrapped harness."""
        return getattr(self._harness, name)


async def benchmark_auto_batching(
    n_iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark auto-batching vs sequential dispatch."""
    print("=" * 70)
    print("AUTO-BATCH DISPATCHER BENCHMARK")
    print("=" * 70)
    
    from aho.mock_tools import long_context_lookup, summarize_report
    
    # Simulated slow dispatch
    async def slow_dispatch(tool_name: str, arguments: dict) -> dict:
        if tool_name == "long_context_lookup":
            await asyncio.sleep(0.1)  # 100ms
            return await long_context_lookup(**arguments)
        elif tool_name == "summarize_report":
            await asyncio.sleep(0.08)  # 80ms
            return await summarize_report(**arguments)
        return {"success": False, "error": "Unknown tool"}
    
    # Create batch dispatcher
    dispatcher = AutoBatchDispatcher(
        original_dispatch=slow_dispatch,
        batch_window_ms=50.0,
    )
    
    sequential_times = []
    batch_times = []
    
    for i in range(n_iterations):
        # Sequential execution (baseline)
        start = time.perf_counter()
        await slow_dispatch("long_context_lookup", {"topic": "climate", "distilled": True})
        await slow_dispatch("summarize_report", {"subject": "carbon", "distilled": True})
        seq_time = (time.perf_counter() - start) * 1000
        sequential_times.append(seq_time)
        
        # Auto-batched execution
        dispatcher.reset_stats()
        start = time.perf_counter()
        
        # Fire both calls quickly (they'll be batched)
        task1 = asyncio.create_task(
            dispatcher.dispatch("long_context_lookup", {"topic": "climate", "distilled": True})
        )
        task2 = asyncio.create_task(
            dispatcher.dispatch("summarize_report", {"subject": "carbon", "distilled": True})
        )
        
        await asyncio.gather(task1, task2)
        batch_time = (time.perf_counter() - start) * 1000
        batch_times.append(batch_time)
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{n_iterations} iterations")
    
    # Calculate stats
    avg_sequential = sum(sequential_times) / len(sequential_times)
    avg_batch = sum(batch_times) / len(batch_times)
    time_saved = avg_sequential - avg_batch
    speedup = avg_sequential / avg_batch if avg_batch > 0 else 0
    
    print(f"\n📊 Results ({n_iterations} iterations):")
    print(f"   Sequential avg: {avg_sequential:.1f}ms")
    print(f"   Auto-batch avg: {avg_batch:.1f}ms")
    print(f"   Time saved: {time_saved:.1f}ms ({(time_saved/avg_sequential)*100:.1f}%)")
    print(f"   Speedup: {speedup:.2f}x")
    
    return {
        "sequential_avg_ms": avg_sequential,
        "batch_avg_ms": avg_batch,
        "time_saved_ms": time_saved,
        "speedup": speedup,
    }


if __name__ == "__main__":
    # Run benchmark
    results = asyncio.run(benchmark_auto_batching(n_iterations=10))
    
    print("\n" + "=" * 70)
    print("AUTO-BATCH DISPATCHER READY")
    print("=" * 70)
