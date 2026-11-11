"""Capture microphone audio and transcribe it with faster-whisper."""

from __future__ import annotations

import logging
import os
import subprocess
from collections import deque
from collections.abc import Callable, Iterator

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad


LOGGER = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
FRAME_DURATION_MS = 30
FRAME_BYTES = SAMPLE_RATE * FRAME_DURATION_MS // 1000 * 2
VAD_WINDOW_FRAMES = 10
VAD_REQUIRED_SPEECH_FRAMES = 8
MIN_SPEECH_FRAMES = 5


class Listener:
    """Read microphone frames and yield transcripts after each utterance."""

    def __init__(
        self,
        model_name: str | None = None,
        chunk_seconds: int | None = None,
        language: str | None = None,
    ) -> None:
        self.max_utterance_seconds = chunk_seconds or int(
            os.getenv("WHISPER_MAX_UTTERANCE_SECONDS", "8")
        )
        self.min_rms = float(os.getenv("WHISPER_MIN_RMS", "0.01"))
        self.vad_threshold = float(
            os.getenv("WHISPER_VAD_THRESHOLD", "0.65")
        )
        LOGGER.info("Loading Silero VAD")
        self._vad = load_silero_vad(onnx=True)
        self.language = language or os.getenv("WHISPER_LANGUAGE", "ja")
        model_name = model_name or os.getenv("WHISPER_MODEL", "small")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        LOGGER.info("Loading Whisper model %s", model_name)
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    def _record_utterance(self) -> np.ndarray:
        # Silero keeps recurrent context between calls; each microphone pass
        # must start with a clean context after the previous utterance ends.
        self._vad.reset_states()
        command = [
            "arecord",
            "--quiet",
            "--format=S16_LE",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            f"--period-size={SAMPLE_RATE * FRAME_DURATION_MS // 1000}",
            f"--buffer-size={SAMPLE_RATE * FRAME_DURATION_MS // 1000 * 2}",
            "--file-type=raw",
            "-",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        max_frames = max(
            round(self.max_utterance_seconds * 1000 / FRAME_DURATION_MS),
            VAD_WINDOW_FRAMES,
        )
        recent_frames: deque[bytes] = deque(maxlen=VAD_WINDOW_FRAMES)
        utterance: list[bytes] = []
        recording = False
        recording_frames = 0
        vad_results: deque[bool] = deque(maxlen=VAD_WINDOW_FRAMES)
        previous_speech_state: bool | None = None

        try:
            while True:
                frame_parts: list[bytes] = []
                while sum(map(len, frame_parts)) < FRAME_BYTES:
                    part = process.stdout.read(FRAME_BYTES - sum(map(len, frame_parts)))
                    if not part:
                        break
                    frame_parts.append(part)
                frame = b"".join(frame_parts)
                if len(frame) < FRAME_BYTES:
                    break
                samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                centered_samples = samples - np.mean(samples)
                rms = float(
                    np.sqrt(np.mean(np.square(centered_samples / 32768.0)))
                )
                vad_frame = np.pad(
                    centered_samples / 32768.0,
                    (0, 512 - len(samples)),
                )
                vad_probability = float(
                    self._vad(torch.from_numpy(vad_frame), SAMPLE_RATE).item()
                )
                vad_detected = vad_probability >= self.vad_threshold
                is_speech = (
                    rms >= self.min_rms
                    and vad_detected
                )
                if is_speech != previous_speech_state:
                    LOGGER.info(
                        "VAD: %s (rms=%.4f, silero=%.3f)",
                        "声になった" if is_speech else "無声になった",
                        rms,
                        vad_probability,
                    )
                    previous_speech_state = is_speech
                recent_frames.append(frame)
                vad_results.append(is_speech)

                if not recording and len(vad_results) == VAD_WINDOW_FRAMES:
                    speech_count = sum(vad_results)
                    if speech_count >= VAD_REQUIRED_SPEECH_FRAMES:
                        recording = True
                        recording_frames = 0
                        utterance.extend(recent_frames)
                        LOGGER.info(
                            "Speech started: %d/%d speech frames",
                            speech_count,
                            VAD_WINDOW_FRAMES,
                        )
                elif recording:
                    utterance.append(frame)
                    recording_frames += 1
                    silence_count = VAD_WINDOW_FRAMES - sum(vad_results)
                    if (
                        recording_frames >= MIN_SPEECH_FRAMES
                        and len(vad_results) == VAD_WINDOW_FRAMES
                        and silence_count >= VAD_REQUIRED_SPEECH_FRAMES
                    ):
                        LOGGER.info(
                            "Speech ended: %d/%d silence frames; captured %.2fs",
                            silence_count,
                            VAD_WINDOW_FRAMES,
                            len(utterance) * FRAME_DURATION_MS / 1000,
                        )
                        break
                    if len(utterance) >= max_frames:
                        LOGGER.warning(
                            "Maximum utterance length reached without VAD end: %.1fs",
                            self.max_utterance_seconds,
                        )
                        break
        finally:
            process.terminate()
            process.wait()

        return np.frombuffer(b"".join(utterance), dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe_once(
        self,
        classify: Callable[[str], str | None] | None = None,
    ) -> tuple[str, str | None]:
        """Record one utterance and classify it as soon as text is available."""
        audio = self._record_utterance()
        if audio.size == 0:
            return "", None
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            without_timestamps=True,
        )
        transcript_parts: list[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            transcript_parts.append(text)
            transcript = " ".join(transcript_parts)
            if classify is not None:
                meaning = classify(transcript)
                if meaning is not None:
                    return transcript, meaning
        return " ".join(transcript_parts), None

    def listen(
        self,
        classify: Callable[[str], str | None] | None = None,
    ) -> Iterator[tuple[str, str]]:
        """Yield recognized commands and immediately resume after irrelevant speech."""
        while True:
            transcript, meaning = self.transcribe_once(classify)
            if meaning is not None:
                yield transcript, meaning