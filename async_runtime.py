"""One long-lived asyncio loop for provider clients that use async HTTP."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import threading
from typing import Coroutine, TypeVar

T = TypeVar("T")


class _PersistentAsyncRuntime:
    """Submit coroutines to one process-wide event loop.

    Async HTTP clients, including Mistral's embedding client, keep connection
    pool state that belongs to the event loop that first uses it.  Creating a
    new loop for every request makes a cached client unsafe to reuse.  A single
    background loop keeps that client valid while still allowing many awaiting
    coroutines to run concurrently.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _start(self) -> asyncio.AbstractEventLoop:
        with self._guard:
            if self._loop is not None and self._loop.is_running():
                return self._loop

            ready = threading.Event()

            def serve() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                loop.run_forever()
                loop.close()

            self._thread = threading.Thread(
                target=serve,
                name="omnirag-async-runtime",
                daemon=True,
            )
            self._thread.start()
            ready.wait()
            assert self._loop is not None
            return self._loop

    def run(self, coroutine: Coroutine[object, object, T]) -> T:
        loop = self._start()
        future: concurrent.futures.Future[T] = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result()

    def stop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)


_runtime = _PersistentAsyncRuntime()
atexit.register(_runtime.stop)


def run(coroutine: Coroutine[object, object, T]) -> T:
    """Run a coroutine on OmniRAG's persistent async runtime."""
    return _runtime.run(coroutine)
