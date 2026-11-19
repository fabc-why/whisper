"""Tests for PcmFrameAssembler."""

from __future__ import annotations

import pytest

from src.openduck_parts.frame_assembler import PcmFrameAssembler


def test_640_bytes_returns_no_frame():
    assembler = PcmFrameAssembler(frame_bytes=960)

    assert assembler.feed(b"\x01" * 640) == []
    assert assembler.buffered_bytes == 640


def test_two_640_byte_chunks_return_one_frame():
    assembler = PcmFrameAssembler(frame_bytes=960)

    assert assembler.feed(b"\x01" * 640) == []
    frames = assembler.feed(b"\x02" * 640)

    assert len(frames) == 1
    assert len(frames[0]) == 960
    assert assembler.buffered_bytes == 320


def test_third_640_byte_chunk_consumes_remainder():
    assembler = PcmFrameAssembler(frame_bytes=960)
    assembler.feed(b"\x01" * 640)
    assembler.feed(b"\x02" * 640)

    frames = assembler.feed(b"\x03" * 640)

    assert len(frames) == 1
    assert len(frames[0]) == 960
    assert assembler.buffered_bytes == 0


def test_1920_bytes_returns_two_frames():
    assembler = PcmFrameAssembler(frame_bytes=960)

    frames = assembler.feed(b"\x5a" * 1920)

    assert len(frames) == 2
    assert all(len(frame) == 960 for frame in frames)
    assert assembler.buffered_bytes == 0


def test_byte_order_is_preserved():
    assembler = PcmFrameAssembler(frame_bytes=960)
    chunk = bytes(index % 256 for index in range(1920))

    frames = assembler.feed(chunk)

    assert [frame for frame in frames] == [
        chunk[:960],
        chunk[960:1920],
    ]
    assert assembler.buffered_bytes == 0


def test_empty_bytes_does_not_change_state():
    assembler = PcmFrameAssembler(frame_bytes=960)
    assembler.feed(b"\x01" * 640)

    assert assembler.feed(b"") == []
    assert assembler.buffered_bytes == 640


def test_clear_removes_remainder():
    assembler = PcmFrameAssembler(frame_bytes=960)
    assembler.feed(b"\x01" * 640)

    assembler.clear()

    assert assembler.buffered_bytes == 0


def test_clear_returns_removed_byte_count():
    assembler = PcmFrameAssembler(frame_bytes=960)
    assembler.feed(b"\x01" * 640)

    assert assembler.clear() == 640
    assert assembler.clear() == 0


@pytest.mark.parametrize("frame_bytes", [0, -1])
def test_non_positive_frame_bytes_is_invalid(frame_bytes):
    with pytest.raises(ValueError):
        PcmFrameAssembler(frame_bytes=frame_bytes)
