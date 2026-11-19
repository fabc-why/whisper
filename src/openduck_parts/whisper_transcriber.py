"""AudioSegment transcription using faster-whisper."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .audio_models import AudioSegment, TranscriptionResult


class WhisperTranscriber:
    """Convert an AudioSegment into a TranscriptionResult."""

    def __init__(
        self,
        model_name: str | None = None,
        language: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_name = (
            model_name
            or os.getenv("WHISPER_MODEL", "small")
        )
        self.language = (
            language
            or os.getenv("WHISPER_LANGUAGE", "ja")
        )
        self.device = (
            device
            or os.getenv("WHISPER_DEVICE", "cpu")
        )
        self.compute_type = (
            compute_type
            or os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        )

        if model is not None:
            self._model = model
        else:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe(
        self,
        segment: AudioSegment,
    ) -> TranscriptionResult:
        """Validate and transcribe one S16LE PCM AudioSegment."""
        if segment.sample_rate != 16_000:
            raise ValueError("AudioSegment sample_rate must be 16000")
        if segment.channels != 1:
            raise ValueError("AudioSegment channels must be 1")
        if len(segment.pcm_bytes) % 2 != 0:
            raise ValueError("S16LE PCM byte length must be even")

        if not segment.pcm_bytes:
            return TranscriptionResult(
                segment_id=segment.segment_id,
                text="",
                language=self.language,
                started_at=segment.started_at,
                ended_at=segment.ended_at,
            )

        audio = (
            np.frombuffer(
                segment.pcm_bytes,
                dtype="<i2",
            ).astype(np.float32)
            / np.float32(32768.0)
        )

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            vad_filter=False,
            without_timestamps=True,
        )

        text_parts: list[str] = []
        for whisper_segment in segments:
            text = whisper_segment.text.strip()
            if text:
                text_parts.append(text)

        detected_language = getattr(info, "language", None)
        if not detected_language:
            detected_language = self.language

        return TranscriptionResult(
            segment_id=segment.segment_id,
            text=" ".join(text_parts),
            language=detected_language,
            started_at=segment.started_at,
            ended_at=segment.ended_at,
        )
