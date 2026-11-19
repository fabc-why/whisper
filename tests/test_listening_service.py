"""Tests for ListeningService assembly."""

from __future__ import annotations

import threading

import pytest

from src.openduck_parts import listening_service


class FakeVadWorker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.completed_segments = 0
        self.dropped_segments = 0

    def start(self):
        self.calls.append("start")

    def stop(self, *, drain=True):
        self.calls.append(("stop", drain))


class FakeWhisperWorker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.processed_segments = 0
        self.empty_transcriptions = 0
        self.failed_transcriptions = 0
        self.callback_failures = 0

    def start(self):
        self.calls.append("start")

    def stop(self, *, drain=True):
        self.calls.append(("stop", drain))


class FakeSubscriber:
    instances = []

    def __init__(self, client, callback):
        self.client = client
        self.callback = callback
        self.calls = []
        self.error = None
        FakeSubscriber.instances.append(self)

    def cleanup(self):
        self.calls.append("cleanup")
        if self.error:
            raise self.error


class FakePublisher:
    def __init__(self):
        self.cleanup_count = 0
        self.cleanup_error = None
        self.published_results = 0
        self.rejected_results = 0
        self.failed_results = 0

    def publish(self, result):
        return True

    def cleanup(self):
        self.cleanup_count += 1
        if self.cleanup_error:
            raise self.cleanup_error


class FakeTranscriber:
    pass


@pytest.fixture
def fake_parts(monkeypatch):
    FakeSubscriber.instances = []
    order = []

    class OrderedVadWorker(FakeVadWorker):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def start(self):
            order.append("vad_start")
            super().start()

        def stop(self, *, drain=True):
            order.append(("vad_stop", drain))
            super().stop(drain=drain)

    class OrderedWhisperWorker(FakeWhisperWorker):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def start(self):
            order.append("whisper_start")
            super().start()

        def stop(self, *, drain=True):
            order.append(("whisper_stop", drain))
            super().stop(drain=drain)

    class OrderedSubscriber(FakeSubscriber):
        def __init__(self, client, callback):
            order.append("subscriber_create")
            super().__init__(client, callback)

        def cleanup(self):
            order.append("subscriber_cleanup")
            super().cleanup()

    monkeypatch.setattr(
        listening_service,
        "VadWorker",
        OrderedVadWorker,
    )
    monkeypatch.setattr(
        listening_service,
        "WhisperWorker",
        OrderedWhisperWorker,
    )
    monkeypatch.setattr(
        listening_service,
        "RawAudioSubscriber",
        OrderedSubscriber,
    )
    return order


def make_service(publisher=None, vad_model=object()):
    return listening_service.ListeningService(
        ros_client=object(),
        vad_model=vad_model,
        transcriber=FakeTranscriber(),
        publisher=publisher or FakePublisher(),
        min_rms=0.01,
        vad_threshold=0.65,
        max_utterance_seconds=8,
    )


def test_start_order(fake_parts):
    service = make_service()

    service.start()

    assert fake_parts == [
        "whisper_start",
        "vad_start",
        "subscriber_create",
    ]


def test_start_twice_does_not_duplicate(fake_parts):
    service = make_service()
    service.start()
    service.start()

    assert fake_parts.count("whisper_start") == 1
    assert fake_parts.count("vad_start") == 1
    assert fake_parts.count("subscriber_create") == 1
    service.stop()


def test_subscriber_callback_is_pcm_queue_put(fake_parts):
    service = make_service()
    service.start()

    subscriber = FakeSubscriber.instances[-1]
    assert subscriber.callback == service.pcm_queue.put


def test_stop_order(fake_parts):
    publisher = FakePublisher()
    service = make_service(publisher)
    service.start()

    service.stop(drain=True)

    assert fake_parts == [
        "whisper_start",
        "vad_start",
        "subscriber_create",
        "subscriber_cleanup",
        ("vad_stop", True),
        ("whisper_stop", True),
    ]
    assert publisher.cleanup_count == 1


@pytest.mark.parametrize("drain", [True, False])
def test_drain_is_passed_to_both_workers(fake_parts, drain):
    service = make_service()
    service.start()

    service.stop(drain=drain)

    assert ("vad_stop", drain) in fake_parts
    assert ("whisper_stop", drain) in fake_parts


def test_stop_twice_is_safe(fake_parts):
    publisher = FakePublisher()
    service = make_service(publisher)
    service.start()

    service.stop()
    service.stop()

    assert publisher.cleanup_count == 1


def test_stop_before_start_is_safe(fake_parts):
    publisher = FakePublisher()
    service = make_service(publisher)
    service.stop()

    assert service.is_running is False
    assert publisher.cleanup_count == 1


def test_startup_failure_rolls_back_workers(fake_parts):
    class FailingSubscriber(FakeSubscriber):
        def __init__(self, client, callback):
            super().__init__(client, callback)
            raise RuntimeError("subscriber failed")

    listening_service.RawAudioSubscriber = FailingSubscriber
    service = make_service()

    with pytest.raises(RuntimeError, match="subscriber failed"):
        service.start()

    assert ("vad_stop", False) in fake_parts
    assert ("whisper_stop", False) in fake_parts
    assert service.is_running is False


def test_cleanup_exception_does_not_stop_remaining_cleanup(fake_parts):
    publisher = FakePublisher()
    publisher.cleanup_error = RuntimeError("publisher cleanup")
    service = make_service(publisher)
    service.start()

    subscriber = FakeSubscriber.instances[-1]
    subscriber.error = RuntimeError("subscriber cleanup")

    service.stop()

    assert ("vad_stop", True) in fake_parts
    assert ("whisper_stop", True) in fake_parts
    assert publisher.cleanup_count == 1


def test_ros_client_is_never_terminated(fake_parts):
    client = type(
        "Client",
        (),
        {"terminate": lambda self: pytest.fail("terminate called")},
    )()
    publisher = FakePublisher()
    service = listening_service.ListeningService(
        client,
        vad_model=object(),
        transcriber=FakeTranscriber(),
        publisher=publisher,
    )
    service.start()

    service.stop()


def test_publisher_publish_is_result_callback(fake_parts):
    publisher = FakePublisher()
    service = make_service(publisher)

    callback = service.whisper_worker.kwargs["result_callback"]
    assert callback == publisher.publish


def test_invalid_numeric_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("WHISPER_MIN_RMS", "bad")

    with pytest.raises(ValueError, match="WHISPER_MIN_RMS"):
        listening_service.ListeningService(
            object(),
            vad_model=object(),
            transcriber=FakeTranscriber(),
            publisher=FakePublisher(),
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("WHISPER_VAD_THRESHOLD", "not-a-number"),
        ("WHISPER_MAX_UTTERANCE_SECONDS", "bad"),
    ],
)
def test_invalid_vad_environment_is_rejected(
    monkeypatch,
    name,
    value,
):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        listening_service.ListeningService(
            object(),
            vad_model=object(),
            transcriber=FakeTranscriber(),
            publisher=FakePublisher(),
        )


def test_is_running_changes(fake_parts):
    service = make_service()

    assert service.is_running is False
    service.start()
    assert service.is_running is True
    service.stop()
    assert service.is_running is False


def test_stop_forbids_restart(fake_parts):
    service = make_service()
    service.start()
    service.stop()

    with pytest.raises(RuntimeError, match="cannot restart"):
        service.start()
