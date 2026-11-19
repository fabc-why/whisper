"""Assemble variable-sized PCM chunks into fixed-size frames."""

from __future__ import annotations


class PcmFrameAssembler:
    """Combine PCM chunks and extract fixed-length frames."""

    def __init__(self, frame_bytes: int = 960) -> None:
        if frame_bytes <= 0:
            raise ValueError("frame_bytes must be greater than zero")

        self.frame_bytes = frame_bytes
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[bytes]:
        """Append a PCM chunk and return all complete fixed-size frames."""
        if not chunk:
            return []

        self._buffer.extend(chunk)
        frames: list[bytes] = []

        while len(self._buffer) >= self.frame_bytes:
            frames.append(bytes(self._buffer[: self.frame_bytes]))
            del self._buffer[: self.frame_bytes]

        return frames

    def clear(self) -> int:
        """Remove and return the number of buffered remainder bytes."""
        removed = len(self._buffer)
        self._buffer.clear()
        return removed

    @property
    def buffered_bytes(self) -> int:
        """Return the number of PCM bytes held for the next frame."""
        return len(self._buffer)
