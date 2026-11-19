"""Tests for TranscriptionPublisher."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.openduck_parts.audio_models import TranscriptionResult


class FakeTopic:
    instances = []

    def __init__(self, client, topic, message_type):
        self.client = client
        self.topic = topic
        self.message_type = message_type
        self.advertise_count = 0
        self.publish_count = 0
        self.unadvertise_count = 0
        self.messages = []
        self.publish_error: Exception | None = None
        self.unadvertise_error: Exception | None = None
        FakeTopic.instances.append(self)

    def advertise(self):
        self.advertise_count += 1

    def publish(self, message):
        self.publish_count += 1
        if self.publish_error is not None:
            raise self.publish_error
        self.messages.append(message)

    def unadvertise(self):
        self.unadvertise_count += 1
        if self.unadvertise_error is not None:
            raise self.unadvertise_error


class FakeMessage:
    def __init__(self, data):
        self.data = data


class FakeRoslibpy:
    Topic = FakeTopic
    Message = FakeMessage


def make_result(text: str = " こっち来て ") -> TranscriptionResult:
    return TranscriptionResult(
        segment_id="segment-123",
        text=text,
        language="ja",
        started_at=datetime(
            2026,
            11,
            19,
            12,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        ended_at=datetime(
            2026,
            11,
            19,
            12,
            0,
            2,
            tzinfo=timezone.utc,
        ),
    )


@pytest.fixture
def publisher_module(monkeypatch):
    import src.openduck_parts.transcription_publisher as module

    FakeTopic.instances = []
    monkeypatch.setattr(module, "roslibpy", FakeRoslibpy)
    return module


def test_initialization_uses_topic_and_advertises(
    publisher_module,
):
    client = object()

    publisher = publisher_module.TranscriptionPublisher(client)

    topic = publisher.topic
    assert topic.topic == publisher_module.TRANSCRIPTION_TOPIC
    assert topic.message_type == "std_msgs/String"
    assert topic.advertise_count == 1
    assert topic.client is client


def test_empty_topic_is_rejected(publisher_module):
    with pytest.raises(ValueError):
        publisher_module.TranscriptionPublisher(
            object(),
            topic="   ",
        )


def test_publish_creates_correct_json(publisher_module):
    publisher = publisher_module.TranscriptionPublisher(object())

    assert publisher.publish(make_result()) is True

    payload = json.loads(publisher.topic.messages[0].data["data"])
    assert payload["segment_id"] == "segment-123"
    assert payload["text"] == "こっち来て"
    assert payload["language"] == "ja"
    assert payload["started_at"] == "2026-11-19T12:00:00+00:00"
    assert payload["ended_at"] == "2026-11-19T12:00:02+00:00"


def test_json_preserves_japanese(publisher_module):
    publisher = publisher_module.TranscriptionPublisher(object())

    publisher.publish(make_result("オープンダック"))

    raw_json = publisher.topic.messages[0].data["data"]
    assert "オープンダック" in raw_json


def test_normal_publish_updates_counter(publisher_module):
    publisher = publisher_module.TranscriptionPublisher(object())

    assert publisher.publish(make_result()) is True
    assert publisher.published_results == 1


def test_empty_text_is_rejected(publisher_module):
    publisher = publisher_module.TranscriptionPublisher(object())

    assert publisher.publish(make_result("   ")) is False
    assert publisher.topic.publish_count == 0
    assert publisher.rejected_results == 1


def test_publish_exception_returns_false(publisher_module):
    publisher = publisher_module.TranscriptionPublisher(object())
    publisher.topic.publish_error = RuntimeError("publish failed")

    assert publisher.publish(make_result()) is False
    assert publisher.failed_results == 1


def test_cleanup_unadvertises_once(publisher_module):
    publisher = publisher_module.TranscriptionPublisher(object())

    publisher.cleanup()
    publisher.cleanup()

    assert publisher.topic.unadvertise_count == 1


def test_publish_after_cleanup_is_rejected(publisher_module):
    publisher = publisher_module.TranscriptionPublisher(object())
    publisher.cleanup()

    assert publisher.publish(make_result()) is False
    assert publisher.rejected_results == 1


def test_cleanup_does_not_terminate_client(publisher_module):
    client = SimpleNamespace(terminate=lambda: pytest.fail(
        "ROS client must not be terminated"
    ))
    publisher = publisher_module.TranscriptionPublisher(client)

    publisher.cleanup()

    assert publisher.topic.unadvertise_count == 1


def test_bound_publish_method_is_usable_as_callback(publisher_module):
    publisher = publisher_module.TranscriptionPublisher(object())

    callback = publisher.publish
    assert callback(make_result()) is True
