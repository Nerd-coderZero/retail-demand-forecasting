"""Micro-batching: coalesce concurrent prediction requests into one model call.

Each request waits at most max_wait_ms for other requests to join its batch. With batching off,
every request is handled on its own, which is the comparison case for the load test.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BatchItem:
    payload: Any
    future: asyncio.Future = field(repr=False)
    queued_at: float = field(default_factory=time.perf_counter)


class Batcher:
    def __init__(self, handler: Callable[[list[Any]], list[Any]], max_batch_size: int, max_wait_ms: int):
        self._handler = handler
        self._max_batch_size = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.on_batch: Callable[[int, float, float], None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, payload: Any) -> Any:
        loop = asyncio.get_running_loop()
        item = BatchItem(payload=payload, future=loop.create_future())
        await self._queue.put(item)
        return await item.future

    async def _collect(self) -> list[BatchItem]:
        first = await self._queue.get()
        items = [first]
        deadline = time.perf_counter() + self._max_wait
        while len(items) < self._max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                items.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return items

    async def _run(self) -> None:
        while True:
            items = await self._collect()
            started = time.perf_counter()
            waited = started - min(item.queued_at for item in items)
            try:
                results = await asyncio.to_thread(self._handler, [item.payload for item in items])
                if len(results) != len(items):
                    raise RuntimeError("handler returned a different number of results than requests")
                for item, result in zip(items, results):
                    if not item.future.done():
                        item.future.set_result(result)
            except Exception as exc:  # one failing batch must not kill the worker
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)
            if self.on_batch is not None:
                self.on_batch(len(items), waited, time.perf_counter() - started)
