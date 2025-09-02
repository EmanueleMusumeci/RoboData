"""
Shared rate limiting for DBpedia API to prevent exceeding the 50 requests/second limit.
This module provides a singleton rate limiter that can be imported and used across
all DBpedia API interactions.
"""
import asyncio
from contextlib import asynccontextmanager
import contextvars

# DBpedia allows 50 requests per second and 100 simultaneous connections
# We'll be extremely conservative to avoid hitting limits
MAX_CONCURRENT_REQUESTS = 10  # Only 1 concurrent request at a time
REQUEST_DELAY = 0.1  # 1 second delay between requests = 1 req/sec max

# Global shared semaphore for all DBpedia API calls
_rate_limit_semaphore = None

# Context variable tracking whether the current context is already inside
# a rate_limited_request. This makes the limiter re-entrant for the same
# async context/task and avoids deadlocks when code creates child tasks
# that themselves call rate_limited_request while the parent still holds
# the semaphore.
_inside_rate_limited: contextvars.ContextVar[bool] = contextvars.ContextVar("_inside_rate_limited", default=False)

def get_rate_limiter():
    """Get the shared rate limiter semaphore."""
    global _rate_limit_semaphore
    if _rate_limit_semaphore is None:
        _rate_limit_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _rate_limit_semaphore

@asynccontextmanager
async def rate_limited_request():
    """
    Context manager for making rate-limited requests to DBpedia.
    Use this to wrap all DBpedia API calls.
    """
    # If we're already inside a rate-limited context in this logical
    # execution flow, don't try to re-acquire the global semaphore.
    # This prevents deadlocks when parent coroutines hold the semaphore
    # and spawn child coroutines that also attempt to acquire it.
    if _inside_rate_limited.get():
        # Already inside rate-limited context for this task/context.
        # We don't await sleep here because the outer context already
        # performed the pacing delay.
        yield
        return

    semaphore = get_rate_limiter()
    # Acquire one slot and mark context so nested calls won't re-acquire.
    async with semaphore:
        token = _inside_rate_limited.set(True)
        try:
            # Enforce a small delay between requests to be conservative.
            await asyncio.sleep(REQUEST_DELAY)
            yield
        finally:
            # Restore previous context value
            _inside_rate_limited.reset(token)
