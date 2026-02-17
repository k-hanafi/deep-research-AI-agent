"""
Concurrency-safe JSONL writer.

Ensures that concurrent asyncio tasks can safely write to the same
JSONL file without interleaved or corrupted lines. Each write is
atomic (locked), flushed immediately for crash safety.

Usage:
    async with AsyncJSONLWriter("output.jsonl") as writer:
        await writer.write({"rcid": 123, "name": "Acme"})
        await writer.write({"rcid": 456, "name": "Beta"})
        print(writer.lines_written)  # 2
"""

import asyncio
import json
from pathlib import Path
from typing import Union


class AsyncJSONLWriter:
    """
    Async-safe JSONL file writer with flush-after-write.

    All writes are serialized through an asyncio.Lock so concurrent
    tasks can call write() safely from asyncio.gather().

    The file is opened in append mode — safe for resume workflows
    where partial data already exists.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._file = None
        self._lines_written = 0

    async def __aenter__(self) -> "AsyncJSONLWriter":
        self._file = open(self.path, "a")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()
            self._file = None

    async def write(self, record: dict) -> None:
        """
        Write a single JSON record as one line, flushed immediately.

        Thread-safe via asyncio.Lock — safe to call from multiple
        concurrent coroutines.
        """
        if self._file is None:
            raise RuntimeError("Writer not open. Use 'async with AsyncJSONLWriter(...)' context.")
        line = json.dumps(record) + "\n"
        async with self._lock:
            self._file.write(line)
            self._file.flush()
            self._lines_written += 1

    @property
    def lines_written(self) -> int:
        """Total lines written during this session."""
        return self._lines_written
