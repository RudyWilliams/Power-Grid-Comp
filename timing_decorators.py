import time
import functools

def runtime_timer(func):
    """Return the runtime of the function"""
    @functools.wraps(func)
    def wrapper_runtime_timer(*args, **kwargs):
        t0 = time.perf_counter()
        value = func(*args, **kwargs)
        t1 = time.perf_counter()
        runtime = t1 - t0
        print(f'{func.__name__} run in: {runtime: 0.4f} seconds')
        return value #so we can still retrieve the data
    return wrapper_runtime_timer

def process_timer(func):
    """Return the CPU time of the function"""
    @functools.wraps(func)
    def wrapper_runtime_timer(*args, **kwargs):
        t0 = time.process_time()
        value = func(*args, **kwargs)
        t1 = time.process_time()
        runtime = t1 - t0
        print(f'{func.__name__} run in: {runtime: 0.4f} CPU seconds')
        return value #so we can still retrieve the data
    return wrapper_runtime_timer