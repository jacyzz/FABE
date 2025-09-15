from __future__ import annotations
import asyncio
from typing import Awaitable, Callable, Iterable, Tuple


async def run_limited(concurrency: int, tasks: Iterable[Callable[[], Awaitable[None]]]) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def runner(task: Callable[[], Awaitable[None]]):
        async with sem:
            await task()

    await asyncio.gather(*(runner(t) for t in tasks))


async def retry(n: int, coro_factory: Callable[[], Awaitable[str]]) -> str:
    last_exc = None
    for i in range(max(1, n)):
        try:
            return await coro_factory()
        except Exception as e:  # pragma: no cover - network/transient
            last_exc = e
            await asyncio.sleep(min(2 ** i, 10))
    raise last_exc  # type: ignore[misc]
