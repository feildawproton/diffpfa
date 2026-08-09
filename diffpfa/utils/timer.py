import os
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Check if profiling is enabled via environment variable
PROFILE_ENABLED = os.environ.get("DIFFPFA_PROFILE", "0") == "1"

@contextmanager
def Timer(name: str):
    """
    A lightweight context manager for performance profiling.
    Only active if DIFFPFA_PROFILE=1 environment variable is set.
    """
    if not PROFILE_ENABLED:
        yield
        return
        
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed = end - start
        print(f"[PROFILE] {name}: {elapsed:.4f} seconds")

def log_time(name: str):
    """
    A decorator for performance profiling functions.
    """
    def decorator(func):
        if not PROFILE_ENABLED:
            return func
            
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = end - start
            print(f"[PROFILE] {name} ({func.__name__}): {elapsed:.4f} seconds")
            return result
        return wrapper
    return decorator
