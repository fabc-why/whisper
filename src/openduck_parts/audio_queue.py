"""Thread-safe queue for incoming PCM audio chunks."""

from __future__ import annotations

import logging
import queue
import threading


LOGGER = logging.getLogger(__name__)


class PcmAudioQueue:
    """Hold PCM byte chunks for transfer to a worker thread."""

    def __init__(self, max_chunks: int = 500) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero")

        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_chunks)
        self._lock = threading.Lock()
        self.received_chunks = 0
        self.dropped_chunks = 0
        self.rejected_chunks = 0

    def put(self, audio: bytes) -> bool:
        """Add a chunk without blocking, replacing the oldest chunk if full."""
        if not audio:
            with self._lock:
                self.rejected_chunks += 1
            return False

        with self._lock:
            try:
                self._queue.put_nowait(audio)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    # A concurrent get() can remove the last chunk.
                    pass
                else:
                    self._queue.task_done()

                try:
                    self._queue.put_nowait(audio)
                except queue.Full:
                    return False

                self.dropped_chunks += 1
                if (
                    self.dropped_chunks == 1
                    or self.dropped_chunks % 100 == 0
                ):
                    LOGGER.warning(
                        "PCM audio queue full; dropped_chunks=%d",
                        self.dropped_chunks,
                    )

            self.received_chunks += 1
            return True

    def get(self, timeout: float = 0.1) -> bytes | None:
        """Get one chunk, returning None when the timeout expires."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Mark a chunk returned by get() as processed."""
        self._queue.task_done()

    def clear(self) -> int:
        """Remove all currently queued chunks and update unfinished_tasks."""
        removed = 0
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()
                removed += 1
        return removed

    @property
    def size(self) -> int:
        """Return the current number of chunks in the queue."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Return whether the queue contains no chunks."""
        return self._queue.empty()
