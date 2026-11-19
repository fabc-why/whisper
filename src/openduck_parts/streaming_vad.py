"""Streaming Silero VAD state machine for fixed-size PCM frames."""

from __future__ import annotations

import logging
import uuid
from collections import deque
from datetime import datetime, timezone

import numpy as np
import torch

from .audio_models import AudioSegment


LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_BYTES = 960
VAD_WINDOW_FRAMES = 10
VAD_REQUIRED_SPEECH_FRAMES = 8
MIN_SPEECH_FRAMES = 5


class StreamingVad:
    """Build one utterance at a time from 30 ms PCM frames."""

    def __init__(
        self,
        vad_model,
        *,
        min_rms: float = 0.01,
        vad_threshold: float = 0.65,
        max_utterance_seconds: int = 8,
    ) -> None:
        self.vad_model = vad_model
        self.min_rms = float(min_rms)
        self.vad_threshold = float(vad_threshold)
        self.max_utterance_seconds = int(max_utterance_seconds)
        self.max_frames = max(
            round(self.max_utterance_seconds * 1000 / FRAME_DURATION_MS),
            VAD_WINDOW_FRAMES,
        )

        self._recent_frames: deque[bytes] = deque(
            maxlen=VAD_WINDOW_FRAMES
        )
        self._vad_results: deque[bool] = deque(
            maxlen=VAD_WINDOW_FRAMES
        )
        self._utterance: list[bytes] = []
        self._recording = False
        self._recording_frames = 0
        self._started_at: datetime | None = None

    @property
    def is_recording(self) -> bool:
        """Whether an utterance is currently being recorded."""
        return self._recording

    def process_frame(self, frame: bytes) -> AudioSegment | None:
        """Process one 960-byte S16LE PCM frame."""
        if len(frame) != FRAME_BYTES:
            raise ValueError(
                f"frame must be exactly {FRAME_BYTES} bytes"
            )

        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
        centered_samples = samples - np.mean(samples)
        rms = float(
            np.sqrt(
                np.mean(np.square(centered_samples / 32768.0))
            )
        )
        vad_frame = centered_samples / 32768.0
        vad_frame = np.pad(vad_frame, (0, 512 - len(vad_frame)))
        vad_result = self.vad_model(
            torch.from_numpy(vad_frame),
            SAMPLE_RATE,
        )
        vad_probability = float(
            vad_result.item()
            if hasattr(vad_result, "item")
            else vad_result
        )
        is_speech = (
            rms >= self.min_rms
            and vad_probability >= self.vad_threshold
        )

        self._recent_frames.append(frame)
        self._vad_results.append(is_speech)

        if not self._recording:
            if len(self._vad_results) == VAD_WINDOW_FRAMES:
                speech_count = sum(self._vad_results)
                if speech_count >= VAD_REQUIRED_SPEECH_FRAMES:
                    self._recording = True
                    self._recording_frames = 0
                    self._started_at = datetime.now(timezone.utc)
                    self._utterance.extend(self._recent_frames)
                    LOGGER.info(
                        "Speech started: %d/%d speech frames",
                        speech_count,
                        VAD_WINDOW_FRAMES,
                    )
            return None

        self._utterance.append(frame)
        self._recording_frames += 1

        silence_count = (
            VAD_WINDOW_FRAMES - sum(self._vad_results)
        )
        if (
            self._recording_frames >= MIN_SPEECH_FRAMES
            and len(self._vad_results) == VAD_WINDOW_FRAMES
            and silence_count >= VAD_REQUIRED_SPEECH_FRAMES
        ):
            LOGGER.info(
                "Speech ended: %d/%d silence frames; captured %.2fs",
                silence_count,
                VAD_WINDOW_FRAMES,
                len(self._utterance) * FRAME_DURATION_MS / 1000,
            )
            return self._finish_segment()

        if len(self._utterance) >= self.max_frames:
            LOGGER.warning(
                "Maximum utterance length reached without VAD end: %.1fs",
                self.max_utterance_seconds,
            )
            return self._finish_segment()

        return None

    def flush(self) -> AudioSegment | None:
        """Return a currently recorded utterance, if one exists."""
        if not self._recording:
            return None

        LOGGER.info(
            "Speech flushed; captured %.2fs",
            len(self._utterance) * FRAME_DURATION_MS / 1000,
        )
        return self._finish_segment()

    def reset(self) -> None:
        """Clear recording state and reset the external VAD model."""
        self._recent_frames.clear()
        self._vad_results.clear()
        self._utterance.clear()
        self._recording = False
        self._recording_frames = 0
        self._started_at = None
        self.vad_model.reset_states()

    def _finish_segment(self) -> AudioSegment:
        """Create and reset a segment for the completed utterance."""
        ended_at = datetime.now(timezone.utc)
        started_at = self._started_at or ended_at
        pcm_bytes = b"".join(self._utterance)
        segment = AudioSegment(
            segment_id=str(uuid.uuid4()),
            pcm_bytes=pcm_bytes,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            started_at=started_at,
            ended_at=ended_at,
        )
        self.reset()
        return segment
