"""Data models for received PCM audio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """One complete speech segment in S16LE PCM format."""

    segment_id: str
    pcm_bytes: bytes
    sample_rate: int
    channels: int
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Text transcription for one complete audio segment."""

    segment_id: str
    text: str
    language: str
    started_at: datetime
    ended_at: datetime
