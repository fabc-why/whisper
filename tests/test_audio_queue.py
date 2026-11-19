"""Tests for PcmAudioQueue."""

from __future__ import annotations

import threading

import pytest

from src.openduck_parts.audio_queue import PcmAudioQueue


def test_put_get_bytes():
    audio_queue = PcmAudioQueue()
    assert audio_queue.put(b"\x01\x02") is True
    assert audio_queue.get(timeout=0.01) == b"\x01\x02"
    audio_queue.task_done()


def test_fifo_order():
    audio_queue = PcmAudioQueue()
    for chunk in (b"first", b"second", b"third"):
        assert audio_queue.put(chunk) is True

    for expected in (b"first", b"second", b"third"):
        assert audio_queue.get(timeout=0.01) == expected
        audio_queue.task_done()


def test_empty_bytes_is_rejected():
    audio_queue = PcmAudioQueue()
    assert audio_queue.put(b"") is False
    assert audio_queue.rejected_chunks == 1
    assert audio_queue.received_chunks == 0


def test_normal_put_increments_received_chunks():
    audio_queue = PcmAudioQueue()
    assert audio_queue.put(b"chunk") is True
    assert audio_queue.received_chunks == 1


def test_get_timeout_returns_none():
    audio_queue = PcmAudioQueue()
    assert audio_queue.get(timeout=0.01) is None


@pytest.mark.parametrize("max_chunks", [0, -1])
def test_non_positive_max_chunks_is_invalid(max_chunks):
    with pytest.raises(ValueError):
        PcmAudioQueue(max_chunks=max_chunks)


def test_full_queue_drops_oldest_and_keeps_latest():
    audio_queue = PcmAudioQueue(max_chunks=2)
    assert audio_queue.put(b"oldest") is True
    assert audio_queue.put(b"second") is True
    assert audio_queue.put(b"latest") is True

    assert audio_queue.dropped_chunks == 1
    assert audio_queue.get(timeout=0.01) == b"second"
    audio_queue.task_done()
    assert audio_queue.get(timeout=0.01) == b"latest"
    audio_queue.task_done()


def test_clear_removes_every_chunk():
    audio_queue = PcmAudioQueue()
    for chunk in (b"a", b"b", b"c"):
        audio_queue.put(chunk)

    assert audio_queue.clear() == 3
    assert audio_queue.size == 0


def test_multiple_threads_put_and_get_without_exceptions():
    audio_queue = PcmAudioQueue(max_chunks=100)
    errors: list[BaseException] = []

    def producer(start: int) -> None:
        try:
            for index in range(start, start + 200):
                audio_queue.put(index.to_bytes(2, "big"))
        except BaseException as error:
            errors.append(error)

    def consumer() -> None:
        try:
            while True:
                chunk = audio_queue.get(timeout=0.01)
                if chunk is None:
                    if audio_queue.size == 0:
                        return
                    continue
                audio_queue.task_done()
        except BaseException as error:
            errors.append(error)

    producers = [
        threading.Thread(target=producer, args=(0,)),
        threading.Thread(target=producer, args=(200,)),
    ]
    consumers = [
        threading.Thread(target=consumer),
        threading.Thread(target=consumer),
    ]
    for thread in producers + consumers:
        thread.start()
    for thread in producers:
        thread.join()

    for _ in range(100):
        chunk = audio_queue.get(timeout=0.01)
        if chunk is None:
            break
        audio_queue.task_done()

    for thread in consumers:
        thread.join(timeout=1)

    assert not errors


def test_task_done_matches_queue_unfinished_tasks():
    audio_queue = PcmAudioQueue()
    for chunk in (b"a", b"b"):
        assert audio_queue.put(chunk) is True
    assert audio_queue._queue.unfinished_tasks == 2

    for expected in (b"a", b"b"):
        assert audio_queue.get(timeout=0.01) == expected
        audio_queue.task_done()
    assert audio_queue._queue.unfinished_tasks == 0
