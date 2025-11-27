# utils.py
import time
from contextlib import contextmanager

@contextmanager
def timer(name="op"):
    t0 = time.time()
    try:
        yield
    finally:
        print(f"[{name}] done in {time.time()-t0:.2f}s")