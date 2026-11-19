"""Tests for WhisperTranscriber."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.openduck_parts.audio_models import (
    AudioSegment,
)
from src.openduck_parts.whisper_transcriber import WhisperTranscriber


def make_segment(
    pcm_bytes: bytes,
    *,
    segment_id: str = "segment-1",
    sample_rate: int = 16_000,
    channels: int = 1,
) -> AudioSegment:
    started = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
    ended = datetime(2026, 11, 19, 12, 0, 1, tzinfo=timezone.utc)
    return AudioSegment(
        segment_id=segment_id,
        pcm_bytes=pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        started_at=started,
        ended_at=ended,
    )


class FakeModel:
    def __init__(
        self,
        whisper_segments=None,
        language: str | None = "ja",
    ) -> None:
        self.whisper_segments = whisper_segments or []
        self.info = (
            SimpleNamespace(language=language)
            if language is not None
            else SimpleNamespace()
        )
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(self.whisper_segments), self.info


def test_pcm_is_converted_to_float32():
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)

    transcriber.transcribe(make_segment(b"\x00\x80\xff\x7f"))

    audio = model.calls[0][0]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32


def test_minimum_pcm_value_is_about_minus_one():
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)

    transcriber.transcribe(make_segment((-32768).to_bytes(2, "little", signed=True)))

    assert model.calls[0][0][0] == pytest.approx(-1.0)


def test_maximum_pcm_value_is_about_point99997():
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)

    transcriber.transcribe(make_segment((32767).to_bytes(2, "little", signed=True)))

    assert model.calls[0][0][0] == pytest.approx(32767 / 32768)


def test_segments_are_joined_and_empty_and_whitespace_removed():
    model = FakeModel([
        SimpleNamespace(text=" オープンダック "),
        SimpleNamespace(text=" "),
        SimpleNamespace(text=" こっち来て "),
    ])
    transcriber = WhisperTranscriber(model=model)

    result = transcriber.transcribe(make_segment(b"\x00\x00\x00\x00"))

    assert result.text == "オープンダック こっち来て"


def test_vad_and_timestamp_options_are_passed():
    model = FakeModel()
    transcriber = WhisperTranscriber(
        model=model,
        language="ja",
    )

    transcriber.transcribe(make_segment(b"\x00\x00"))

    kwargs = model.calls[0][1]
    assert kwargs["vad_filter"] is False
    assert kwargs["without_timestamps"] is True
    assert kwargs["language"] == "ja"


def test_segment_identity_and_time_are_inherited():
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)
    segment = make_segment(b"\x00\x00", segment_id="audio-123")

    result = transcriber.transcribe(segment)

    assert result.segment_id == "audio-123"
    assert result.started_at == segment.started_at
    assert result.ended_at == segment.ended_at


def test_info_language_is_used():
    model = FakeModel(language="en")
    transcriber = WhisperTranscriber(model=model, language="ja")

    result = transcriber.transcribe(make_segment(b"\x00\x00"))

    assert result.language == "en"


def test_missing_info_language_uses_configured_language():
    model = FakeModel(language=None)
    transcriber = WhisperTranscriber(model=model, language="ja")

    result = transcriber.transcribe(make_segment(b"\x00\x00"))

    assert result.language == "ja"


def test_empty_pcm_does_not_call_model():
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)
    segment = make_segment(b"")

    result = transcriber.transcribe(segment)

    assert result.text == ""
    assert model.calls == []


def test_odd_pcm_byte_length_is_rejected():
    transcriber = WhisperTranscriber(model=FakeModel())

    with pytest.raises(ValueError):
        transcriber.transcribe(make_segment(b"\x00"))


def test_invalid_sample_rate_is_rejected():
    transcriber = WhisperTranscriber(model=FakeModel())

    with pytest.raises(ValueError):
        transcriber.transcribe(make_segment(b"\x00\x00", sample_rate=8000))


def test_invalid_channels_are_rejected():
    transcriber = WhisperTranscriber(model=FakeModel())

    with pytest.raises(ValueError):
        transcriber.transcribe(make_segment(b"\x00\x00", channels=2))


def test_model_exception_is_not_swallowed():
    class RaisingModel:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("transcription failed")

    transcriber = WhisperTranscriber(model=RaisingModel())

    with pytest.raises(RuntimeError, match="transcription failed"):
        transcriber.transcribe(make_segment(b"\x00\x00"))


def test_injected_model_is_used_without_replacement():
    model = FakeModel()
    transcriber = WhisperTranscriber(model=model)

    assert transcriber._model is model
