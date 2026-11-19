"""Tests for WhisperWorker."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

from src.openduck_parts.audio_models import (
    AudioSegment,
    TranscriptionResult,
)
from src.openduck_parts.whisper_worker import WhisperWorker


def make_segment(index: int = 1) -> AudioSegment:
    now = datetime.now(timezone.utc)
    return AudioSegment(
        segment_id=str(index),
        pcm_bytes=b"\x00\x00",
        sample_rate=16000,
        channels=1,
        started_at=now,
        ended_at=now,
    )


def make_result(segment: AudioSegment, text: str) -> TranscriptionResult:
    return TranscriptionResult(
        segment_id=segment.segment_id,
        text=text,
        language="ja",
        started_at=segment.started_at,
        ended_at=segment.ended_at,
    )


class FakeVadWorker:
    def __init__(self, segments=None) -> None:
        self.segments = list(segments or [])
        self.task_done_count = 0
        self.clear_count = 0

    @property
    def segment_queue_empty(self) -> bool:
        return not self.segments

    def get_segment(self, timeout: float = 0.1):
        if self.segments:
            return self.segments.pop(0)
        return None

    def segment_task_done(self) -> None:
        self.task_done_count += 1

    def clear_segments(self) -> int:
        removed = len(self.segments)
        self.segments.clear()
        self.clear_count += 1
        return removed


class FakeTranscriber:
    def __init__(self) -> None:
        self.segments = []
        self.results = {}
        self.fail_ids = set()
        self.block_id: str | None = None
        self.started = threading.Event()
        self.release = threading.Event()

    def transcribe(self, segment: AudioSegment):
        self.segments.append(segment)
        if segment.segment_id in self.fail_ids:
            raise RuntimeError("fake transcription failure")
        if segment.segment_id == self.block_id:
            self.started.set()
            if not self.release.wait(timeout=1.0):
                raise RuntimeError("timed out waiting")
        return self.results.get(
            segment.segment_id,
            make_result(segment, ""),
        )


def test_start_starts_worker_thread():
    vad = FakeVadWorker()
    worker = WhisperWorker(vad, FakeTranscriber(), lambda result: None)

    worker.start()
    try:
        assert worker.is_running
        assert worker._thread is not None
        assert worker._thread.name == "openduck-whisper-worker"
        assert worker._thread.daemon is False
    finally:
        worker.stop(drain=False)


def test_start_twice_does_not_duplicate_thread():
    vad = FakeVadWorker()
    worker = WhisperWorker(vad, FakeTranscriber(), lambda result: None)
    worker.start()
    first_thread = worker._thread

    worker.start()

    assert worker._thread is first_thread
    worker.stop(drain=False)


def test_segment_is_passed_to_transcriber_and_callback():
    segment = make_segment(1)
    vad = FakeVadWorker([segment])
    transcriber = FakeTranscriber()
    transcriber.results["1"] = make_result(segment, " hello ")
    received = []
    worker = WhisperWorker(vad, transcriber, received.append)
    worker.start()

    result = worker._thread is not None
    for _ in range(100):
        if received:
            break
        threading.Event().wait(0.005)

    assert result
    assert transcriber.segments == [segment]
    assert received[0].text == " hello "
    assert vad.task_done_count == 1
    assert worker.processed_segments == 1
    worker.stop(drain=True)


def test_empty_text_skips_callback_and_increments_empty():
    segment = make_segment(1)
    vad = FakeVadWorker([segment])
    transcriber = FakeTranscriber()
    transcriber.results["1"] = make_result(segment, "   ")
    callback_called = []
    worker = WhisperWorker(vad, transcriber, callback_called.append)
    worker.start()

    for _ in range(100):
        if worker.empty_transcriptions:
            break
        threading.Event().wait(0.005)

    assert callback_called == []
    assert worker.empty_transcriptions == 1
    worker.stop(drain=True)


def test_transcriber_exception_is_counted_and_worker_continues():
    first = make_segment(1)
    second = make_segment(2)
    vad = FakeVadWorker([first, second])
    transcriber = FakeTranscriber()
    transcriber.fail_ids.add("1")
    transcriber.results["2"] = make_result(second, "second")
    received = []
    worker = WhisperWorker(vad, transcriber, received.append)
    worker.start()

    for _ in range(100):
        if received:
            break
        threading.Event().wait(0.005)

    assert worker.failed_transcriptions == 1
    assert received[0].segment_id == "2"
    assert vad.task_done_count == 2
    worker.stop(drain=True)


def test_callback_exception_is_counted_and_worker_continues():
    first = make_segment(1)
    second = make_segment(2)
    vad = FakeVadWorker([first, second])
    transcriber = FakeTranscriber()
    transcriber.results["1"] = make_result(first, "first")
    transcriber.results["2"] = make_result(second, "second")

    def callback(result):
        if result.segment_id == "1":
            raise RuntimeError("callback failure")

    received = []
    worker = WhisperWorker(
        vad,
        transcriber,
        lambda result: (
            callback(result)
            if result.segment_id == "1"
            else received.append(result)
        ),
    )
    worker.start()

    for _ in range(100):
        if received:
            break
        threading.Event().wait(0.005)

    assert worker.callback_failures == 1
    assert received[0].segment_id == "2"
    assert vad.task_done_count == 2
    worker.stop(drain=True)


def test_drain_true_processes_remaining_segments():
    segment = make_segment(1)
    vad = FakeVadWorker([segment])
    transcriber = FakeTranscriber()
    transcriber.results["1"] = make_result(segment, "drained")
    received = []
    worker = WhisperWorker(vad, transcriber, received.append)
    worker.start()

    worker.stop(drain=True)

    assert received
    assert vad.task_done_count == 1


def test_drain_false_clears_and_does_not_process_segments():
    segment = make_segment(1)
    vad = FakeVadWorker()
    transcriber = FakeTranscriber()
    worker = WhisperWorker(vad, transcriber, lambda result: None)
    worker.start()

    worker.stop(drain=False)
    vad.segments.append(segment)

    assert vad.clear_count == 1
    assert transcriber.segments == []


def test_stop_twice_is_safe():
    vad = FakeVadWorker()
    worker = WhisperWorker(vad, FakeTranscriber(), lambda result: None)
    worker.start()

    worker.stop(drain=False)
    worker.stop(drain=False)

    assert worker.is_running is False


def test_stop_before_start_is_safe():
    worker = WhisperWorker(
        FakeVadWorker(),
        FakeTranscriber(),
        lambda result: None,
    )

    worker.stop(drain=True)
    worker.stop(drain=False)

    assert worker.is_running is False


def test_stop_after_thread_start_waits_for_thread():
    vad = FakeVadWorker()
    worker = WhisperWorker(vad, FakeTranscriber(), lambda result: None)
    worker.start()

    worker.stop(drain=False)

    assert worker._thread is not None
    assert worker._thread.is_alive() is False


def test_stop_waits_for_current_transcription_to_finish():
    segment = make_segment(1)
    vad = FakeVadWorker([segment])
    transcriber = FakeTranscriber()
    transcriber.block_id = "1"
    transcriber.results["1"] = make_result(segment, "finished")
    received = []
    worker = WhisperWorker(vad, transcriber, received.append)
    worker.start()
    assert transcriber.started.wait(timeout=1.0)

    worker.stop(drain=True, timeout=0.05)
    transcriber.release.set()
    worker.stop(drain=True, timeout=1.0)

    assert received
    assert worker.is_running is False
