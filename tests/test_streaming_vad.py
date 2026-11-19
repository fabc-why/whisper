"""Tests for StreamingVad."""

from __future__ import annotations

from datetime import timezone
from uuid import UUID

import pytest
import torch

from src.openduck_parts.audio_models import AudioSegment
from src.openduck_parts.streaming_vad import (
    FRAME_BYTES,
    StreamingVad,
)


class FakeVadModel:
    def __init__(self, probability: float = 0.0) -> None:
        self.probability = probability
        self.reset_count = 0

    def __call__(self, frame, sample_rate: int):
        return torch.tensor([self.probability], dtype=torch.float32)

    def reset_states(self) -> None:
        self.reset_count += 1


def pcm_frame(pattern: int = 1, frames: int = 480) -> bytes:
    values = [
        (pattern * 1000 + index * 100) % 24000 - 12000
        for index in range(frames)
    ]
    return b"".join(
        int(value).to_bytes(2, "little", signed=True)
        for value in values
    )


def test_rejects_frame_with_wrong_length():
    vad = StreamingVad(FakeVadModel())

    with pytest.raises(ValueError):
        vad.process_frame(b"\x00" * (FRAME_BYTES - 1))


def test_silence_returns_no_segment():
    vad = StreamingVad(FakeVadModel(probability=0.0))

    for _ in range(20):
        assert vad.process_frame(pcm_frame()) is None

    assert vad.is_recording is False


def test_speech_start_sets_recording():
    vad = StreamingVad(FakeVadModel(probability=0.9))

    for _ in range(10):
        assert vad.process_frame(pcm_frame()) is None

    assert vad.is_recording is True


def test_preroll_is_included_in_segment():
    vad = StreamingVad(FakeVadModel(probability=0.9))

    for _ in range(10):
        vad.process_frame(pcm_frame())

    vad.vad_model.probability = 0.0
    segment = None
    for _ in range(10):
        segment = vad.process_frame(pcm_frame(2))
        if segment is not None:
            break

    assert segment is not None
    assert len(segment.pcm_bytes) >= 10 * FRAME_BYTES


def test_silence_after_speech_returns_segment():
    vad = StreamingVad(FakeVadModel(probability=0.9))
    for _ in range(10):
        vad.process_frame(pcm_frame())

    vad.vad_model.probability = 0.0
    segment = None
    for _ in range(10):
        segment = vad.process_frame(pcm_frame(2))
        if segment is not None:
            break

    assert isinstance(segment, AudioSegment)
    assert vad.is_recording is False


def test_segment_pcm_order_is_preserved():
    vad = StreamingVad(FakeVadModel(probability=0.9))
    first_frame = pcm_frame(1)
    for _ in range(10):
        vad.process_frame(first_frame)

    vad.vad_model.probability = 0.0
    second_frame = pcm_frame(2)
    segment = None
    for _ in range(10):
        segment = vad.process_frame(second_frame)
        if segment is not None:
            break

    assert segment is not None
    assert segment.pcm_bytes.startswith(first_frame)
    assert second_frame in segment.pcm_bytes


def test_segment_has_uuid_and_utc_datetimes():
    vad = StreamingVad(FakeVadModel(probability=0.9))
    for _ in range(10):
        vad.process_frame(pcm_frame())
    vad.vad_model.probability = 0.0

    segment = None
    for _ in range(10):
        segment = vad.process_frame(pcm_frame(2))
        if segment is not None:
            break

    assert segment is not None
    UUID(segment.segment_id)
    assert segment.started_at.tzinfo is not None
    assert segment.ended_at.tzinfo is not None
    assert segment.started_at.utcoffset() == timezone.utc.utcoffset(None)
    assert segment.ended_at.utcoffset() == timezone.utc.utcoffset(None)


def test_reset_states_is_called_after_segment_end():
    vad_model = FakeVadModel(probability=0.9)
    vad = StreamingVad(vad_model)
    for _ in range(10):
        vad.process_frame(pcm_frame())

    vad_model.probability = 0.0
    segment = None
    for _ in range(10):
        segment = vad.process_frame(pcm_frame(2))
        if segment is not None:
            break

    assert segment is not None
    assert vad_model.reset_count == 1


def test_segment_ends_after_maximum_time():
    vad = StreamingVad(
        FakeVadModel(probability=0.9),
        max_utterance_seconds=1,
    )
    for _ in range(10):
        vad.process_frame(pcm_frame())

    segment = None
    for _ in range(50):
        segment = vad.process_frame(pcm_frame(2))
        if segment is not None:
            break

    assert segment is not None


def test_flush_returns_recording_segment():
    vad = StreamingVad(FakeVadModel(probability=0.9))
    for _ in range(10):
        vad.process_frame(pcm_frame())

    segment = vad.flush()

    assert segment is not None
    assert vad.is_recording is False


def test_flush_silence_returns_none():
    vad = StreamingVad(FakeVadModel(probability=0.0))

    assert vad.flush() is None


def test_reset_clears_internal_state():
    vad = StreamingVad(FakeVadModel(probability=0.9))
    for _ in range(10):
        vad.process_frame(pcm_frame())

    vad.reset()

    assert vad.is_recording is False
    assert vad.flush() is None


def test_start_frame_is_not_added_twice():
    vad = StreamingVad(FakeVadModel(probability=0.9))
    for _ in range(10):
        vad.process_frame(pcm_frame(1))

    vad.vad_model.probability = 0.0
    segment = None
    for _ in range(10):
        segment = vad.process_frame(pcm_frame(2))
        if segment is not None:
            break

    assert segment is not None
    assert segment.pcm_bytes.startswith(pcm_frame(1))
    assert len(segment.pcm_bytes) == 18 * FRAME_BYTES
