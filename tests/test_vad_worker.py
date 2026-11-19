"""Tests for the VadWorker thread."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from src.openduck_parts.audio_models import AudioSegment
from src.openduck_parts.audio_queue import PcmAudioQueue
from src.openduck_parts.vad_worker import VadWorker


def make_segment(index: int = 1) -> AudioSegment:
    now = datetime.now(timezone.utc)
    return AudioSegment(
        segment_id=str(index),
        pcm_bytes=b"\x01\x02",
        sample_rate=16000,
        channels=1,
        started_at=now,
        ended_at=now,
    )


class FakeAssembler:
    def __init__(self, frames=None) -> None:
        self.frames = frames or []
        self.chunks: list[bytes] = []
        self.clear_count = 0
        self.feed_event = threading.Event()

    def feed(self, chunk: bytes) -> list[bytes]:
        self.chunks.append(chunk)
        self.feed_event.set()
        return self.frames

    def clear(self) -> int:
        self.clear_count += 1
        return 0


class FakeVad:
    def __init__(self, segment=None) -> None:
        self.segment = segment
        self.frames: list[bytes] = []
        self.process_event = threading.Event()
        self.flush_count = 0
        self.reset_count = 0
        self.raise_for_frame: bytes | None = None

    def process_frame(self, frame: bytes):
        self.frames.append(frame)
        self.process_event.set()
        if frame == self.raise_for_frame:
            raise RuntimeError("test frame failure")
        return self.segment

    def flush(self):
        self.flush_count += 1
        return self.segment

    def reset(self) -> None:
        self.reset_count += 1


class TrackingPcmQueue(PcmAudioQueue):
    def __init__(self) -> None:
        super().__init__(max_chunks=100)
        self.task_done_count = 0

    def task_done(self) -> None:
        self.task_done_count += 1
        super().task_done()


def wait_for_event(event: threading.Event) -> None:
    assert event.wait(timeout=1.0)


def test_start_starts_worker_thread():
    pcm_queue = PcmAudioQueue()
    worker = VadWorker(pcm_queue, FakeAssembler(), FakeVad())

    worker.start()
    try:
        assert worker.is_running
        assert worker._thread is not None
        assert worker._thread.name == "openduck-vad-worker"
        assert worker._thread.daemon is False
    finally:
        worker.stop(drain=False)


def test_start_twice_does_not_duplicate_thread():
    pcm_queue = PcmAudioQueue()
    worker = VadWorker(pcm_queue, FakeAssembler(), FakeVad())

    worker.start()
    first_thread = worker._thread
    worker.start()

    assert worker._thread is first_thread
    worker.stop(drain=False)


def test_pcm_chunk_is_fed_and_multiple_frames_are_ordered():
    pcm_queue = PcmAudioQueue()
    assembler = FakeAssembler([b"a" * 960, b"b" * 960])
    vad = FakeVad()
    worker = VadWorker(pcm_queue, assembler, vad)

    pcm_queue.put(b"chunk")
    worker.start()
    wait_for_event(vad.process_event)

    assert assembler.chunks == [b"chunk"]
    assert vad.frames == [b"a" * 960, b"b" * 960]
    worker.stop(drain=False)


def test_process_none_adds_no_segment():
    pcm_queue = PcmAudioQueue()
    worker = VadWorker(
        pcm_queue,
        FakeAssembler([b"a" * 960]),
        FakeVad(segment=None),
    )
    pcm_queue.put(b"chunk")
    worker.start()

    for _ in range(100):
        if worker._frame_assembler.chunks:
            break
    assert worker.get_segment(timeout=0.01) is None
    worker.stop(drain=False)


def test_segment_is_enqueued_and_completed_count_increments():
    pcm_queue = PcmAudioQueue()
    segment = make_segment()
    vad = FakeVad(segment=segment)
    worker = VadWorker(
        pcm_queue,
        FakeAssembler([b"a" * 960]),
        vad,
    )
    pcm_queue.put(b"chunk")
    worker.start()

    received = worker.get_segment(timeout=1.0)
    assert received is segment
    assert worker.completed_segments == 1
    worker.segment_task_done()
    worker.stop(drain=False)


def test_pcm_chunk_task_done_is_called_once():
    pcm_queue = TrackingPcmQueue()
    assembler = FakeAssembler([b"a" * 960])
    worker = VadWorker(pcm_queue, assembler, FakeVad())
    pcm_queue.put(b"chunk")
    worker.start()
    wait_for_event(assembler.feed_event)

    pcm_queue._queue.join()
    assert pcm_queue.task_done_count == 1
    worker.stop(drain=False)


def test_segment_queue_full_drops_new_segment():
    pcm_queue = PcmAudioQueue()
    worker = VadWorker(
        pcm_queue,
        FakeAssembler(),
        FakeVad(),
        segment_queue_size=1,
    )
    existing = make_segment(1)
    worker._segment_queue.put(existing)

    assert worker._enqueue_segment(make_segment(2)) is False
    assert worker.dropped_segments == 1
    assert worker.completed_segments == 0
    assert worker.get_segment(timeout=0.01) is existing
    worker.segment_task_done()


def test_get_segment_timeout_and_task_done():
    pcm_queue = PcmAudioQueue()
    worker = VadWorker(pcm_queue, FakeAssembler(), FakeVad())

    assert worker.get_segment(timeout=0.01) is None
    worker._segment_queue.put(make_segment())
    assert worker.get_segment(timeout=0.01) is not None
    worker.segment_task_done()


def test_drain_true_processes_remaining_and_flushes_once():
    pcm_queue = PcmAudioQueue()
    assembler = FakeAssembler([b"a" * 960])
    segment = make_segment()
    vad = FakeVad(segment=segment)
    worker = VadWorker(pcm_queue, assembler, vad)

    pcm_queue.put(b"remaining")
    worker.start()
    worker.stop(drain=True)

    assert assembler.chunks == [b"remaining"]
    assert assembler.clear_count == 1
    assert vad.flush_count == 1
    assert worker.completed_segments >= 1
    assert worker.get_segment(timeout=0.01) is segment


def test_drain_false_clears_queue_assembler_and_resets_vad():
    pcm_queue = PcmAudioQueue()
    pcm_queue.put(b"discard")
    assembler = FakeAssembler()
    vad = FakeVad()
    worker = VadWorker(pcm_queue, assembler, vad)

    worker.start()
    worker.stop(drain=False)

    assert pcm_queue.empty
    assert assembler.clear_count == 1
    assert vad.reset_count == 1
    assert vad.flush_count == 0


def test_stop_twice_is_safe_and_reset_once():
    pcm_queue = PcmAudioQueue()
    assembler = FakeAssembler()
    vad = FakeVad()
    worker = VadWorker(pcm_queue, assembler, vad)

    worker.start()
    worker.stop(drain=False)
    worker.stop(drain=False)

    assert vad.reset_count == 1
    assert worker.is_running is False


def test_stop_without_start_is_safe():
    worker = VadWorker(PcmAudioQueue(), FakeAssembler(), FakeVad())
    worker.stop(drain=False)
    worker.stop(drain=True)

    assert worker.is_running is False


def test_frame_exception_does_not_stop_later_chunk():
    pcm_queue = PcmAudioQueue()
    assembler = FakeAssembler()
    vad = FakeVad()
    vad.raise_for_frame = b"bad"
    worker = VadWorker(pcm_queue, assembler, vad)

    assembler.frames = [b"bad"]
    pcm_queue.put(b"bad")

    def second_chunk() -> None:
        assembler.frames = [b"good"]
        pcm_queue.put(b"good")

    assembler.feed = lambda chunk: (
        [b"bad"] if chunk == b"bad" else [b"good"]
    )
    worker.start()
    second_chunk()
    worker.stop(drain=True)

    assert b"good" in vad.frames


def test_stop_joins_thread_and_thread_is_not_running():
    pcm_queue = PcmAudioQueue()
    worker = VadWorker(pcm_queue, FakeAssembler(), FakeVad())
    worker.start()

    worker.stop(drain=False)

    assert worker._thread is not None
    assert worker._thread.is_alive() is False


def test_stop_does_not_wait_for_vad_processing_lock():
    pcm_queue = PcmAudioQueue()
    feed_started = threading.Event()
    feed_release = threading.Event()

    class BlockingAssembler(FakeAssembler):
        def feed(self, chunk: bytes) -> list[bytes]:
            feed_started.set()
            assert feed_release.wait(timeout=1.0)
            return [b"a" * 960]

    worker = VadWorker(
        pcm_queue,
        BlockingAssembler(),
        FakeVad(),
    )
    pcm_queue.put(b"chunk")
    worker.start()
    assert feed_started.wait(timeout=1.0)

    worker.stop(drain=True, timeout=0.05)

    feed_release.set()
    worker.stop(drain=True, timeout=1.0)
