"""Bounded, ordered process work; scientific identity never depends on scheduling."""

from __future__ import annotations

import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice

from threadpoolctl import threadpool_limits

_LIMITER = None


def initialise_worker(threads):
    global _LIMITER
    if threads:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(threads)
        _LIMITER = threadpool_limits(limits=threads)


def ordered_map(function, tasks, *, workers=1, threads=1, io=False):
    """Keep at most two tasks per worker queued and propagate every failure."""
    if workers < 1 or threads < 0:
        raise ValueError("workers must be positive and threads nonnegative")
    if workers == 1:
        with threadpool_limits(limits=threads or None):
            yield from map(function, tasks)
        return
    kwargs = {} if io else {
        "mp_context": multiprocessing.get_context("spawn"),
        "initializer": initialise_worker, "initargs": (threads,),
    }
    executor = ThreadPoolExecutor if io else ProcessPoolExecutor
    with executor(max_workers=workers, **kwargs) as pool:
        tasks = iter(tasks)
        pending = deque(pool.submit(function, task) for task in islice(tasks, 2 * workers))
        while pending:
            yield pending.popleft().result()
            try:
                task = next(tasks)
            except StopIteration:
                continue
            pending.append(pool.submit(function, task))
