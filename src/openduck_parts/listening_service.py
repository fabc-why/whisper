"""Compose ROS audio intake, VAD, Whisper, and transcription publishing."""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

try:
    from .audio_input import RawAudioSubscriber
    from .audio_queue import PcmAudioQueue
    from .frame_assembler import PcmFrameAssembler
    from .streaming_vad import StreamingVad
    from .transcription_publisher import TranscriptionPublisher
    from .vad_worker import VadWorker
    from .whisper_transcriber import WhisperTranscriber
    from .whisper_worker import WhisperWorker
except ImportError:
    from audio_input import RawAudioSubscriber
    from audio_queue import PcmAudioQueue
    from frame_assembler import PcmFrameAssembler
    from streaming_vad import StreamingVad
    from transcription_publisher import TranscriptionPublisher
    from vad_worker import VadWorker
    from whisper_transcriber import WhisperTranscriber
    from whisper_worker import WhisperWorker


LOGGER = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


class ListeningService:
    """Start and stop the complete external microphone listening pipeline."""

    def __init__(
        self,
        ros_client: Any,
        *,
        model_name: str | None = None,
        language: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        pcm_queue_size: int = 150,
        segment_queue_size: int = 10,
        min_rms: float | None = None,
        vad_threshold: float | None = None,
        max_utterance_seconds: int | None = None,
        vad_model: Any | None = None,
        transcriber: WhisperTranscriber | None = None,
        publisher: TranscriptionPublisher | None = None,
    ) -> None:
        if min_rms is None:
            resolved_min_rms = _env_float(
                "WHISPER_MIN_RMS",
                0.01,
            )
        else:
            resolved_min_rms = float(min_rms)
        if not math.isfinite(resolved_min_rms) or resolved_min_rms < 0:
            raise ValueError(
                "min_rms must be a finite value greater than or equal to zero"
            )

        if vad_threshold is None:
            resolved_vad_threshold = _env_float(
                "WHISPER_VAD_THRESHOLD",
                0.65,
            )
        else:
            resolved_vad_threshold = float(vad_threshold)
        if (
            not math.isfinite(resolved_vad_threshold)
            or not 0.0 <= resolved_vad_threshold <= 1.0
        ):
            raise ValueError(
                "vad_threshold must be between 0.0 and 1.0"
            )

        if max_utterance_seconds is None:
            resolved_max_seconds = _env_int(
                "WHISPER_MAX_UTTERANCE_SECONDS",
                8,
            )
        else:
            resolved_max_seconds = max_utterance_seconds
        if resolved_max_seconds <= 0:
            raise ValueError(
                "max_utterance_seconds must be greater than zero"
            )

        self.ros_client = ros_client
        self.pcm_queue = PcmAudioQueue(
            max_chunks=pcm_queue_size,
        )
        self.frame_assembler = PcmFrameAssembler(
            frame_bytes=960,
        )

        if vad_model is None:
            from silero_vad import load_silero_vad

            vad_model = load_silero_vad(onnx=True)
        self.streaming_vad = StreamingVad(
            vad_model,
            min_rms=resolved_min_rms,
            vad_threshold=resolved_vad_threshold,
            max_utterance_seconds=resolved_max_seconds,
        )
        self.vad_worker = VadWorker(
            pcm_queue=self.pcm_queue,
            frame_assembler=self.frame_assembler,
            streaming_vad=self.streaming_vad,
            segment_queue_size=segment_queue_size,
        )

        self.transcriber = transcriber or WhisperTranscriber(
            model_name=model_name,
            language=language,
            device=device,
            compute_type=compute_type,
        )
        self.transcription_publisher = (
            publisher
            or TranscriptionPublisher(client=ros_client)
        )
        self.whisper_worker = WhisperWorker(
            vad_worker=self.vad_worker,
            transcriber=self.transcriber,
            result_callback=self.transcription_publisher.publish,
        )

        self.audio_subscriber: RawAudioSubscriber | None = None
        self._started = False
        self._stopped = False
        self._state_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Return whether the service is currently running."""
        with self._state_lock:
            return self._started and not self._stopped

    def start(self) -> None:
        """Start downstream consumers before ROS audio subscription."""
        with self._state_lock:
            if self._started:
                if self._stopped:
                    raise RuntimeError(
                        "ListeningService cannot restart after stop"
                    )
                return
            if self._stopped:
                raise RuntimeError(
                    "ListeningService cannot restart after stop"
                )

        try:
            self.whisper_worker.start()
            self.vad_worker.start()
            self.audio_subscriber = RawAudioSubscriber(
                client=self.ros_client,
                callback=self.pcm_queue.put,
            )
        except Exception:
            try:
                self.vad_worker.stop(drain=False)
            except Exception:
                LOGGER.exception(
                    "Failed to stop VadWorker during startup rollback"
                )
            try:
                self.whisper_worker.stop(drain=False)
            except Exception:
                LOGGER.exception(
                    "Failed to stop WhisperWorker during startup rollback"
                )
            try:
                if hasattr(
                    self.transcription_publisher,
                    "cleanup",
                ):
                    self.transcription_publisher.cleanup()
            except Exception:
                LOGGER.exception(
                    "Failed to clean publisher during startup rollback"
                )

            with self._state_lock:
                self._stopped = True
            raise

        with self._state_lock:
            self._started = True
            self._stopped = False

    def stop(self, *, drain: bool = True) -> None:
        """Stop every pipeline stage in downstream-safe order."""
        with self._state_lock:
            if not self._started:
                if not self._stopped:
                    self._stopped = True
                    try:
                        self.transcription_publisher.cleanup()
                    except Exception:
                        LOGGER.exception(
                            "Failed to clean TranscriptionPublisher "
                            "before service start"
                        )
                return
            if self._stopped:
                return
            self._stopped = True

        first_error: Exception | None = None

        try:
            if self.audio_subscriber is not None:
                self.audio_subscriber.cleanup()
        except Exception as error:
            first_error = error
            LOGGER.exception(
                "Failed to clean RawAudioSubscriber"
            )

        try:
            self.vad_worker.stop(drain=drain)
        except Exception as error:
            if first_error is None:
                first_error = error
            LOGGER.exception("Failed to stop VadWorker")

        try:
            self.whisper_worker.stop(drain=drain)
        except Exception as error:
            if first_error is None:
                first_error = error
            LOGGER.exception("Failed to stop WhisperWorker")

        try:
            self.transcription_publisher.cleanup()
        except Exception as error:
            if first_error is None:
                first_error = error
            LOGGER.exception("Failed to clean TranscriptionPublisher")

        LOGGER.info(
            "ListeningService stopped: "
            "pcm_received=%d pcm_dropped=%d pcm_rejected=%d "
            "vad_completed=%d vad_dropped=%d "
            "whisper_processed=%d whisper_empty=%d "
            "whisper_failed=%d whisper_callback_failures=%d "
            "published=%d publish_rejected=%d publish_failed=%d",
            self.pcm_queue.received_chunks,
            self.pcm_queue.dropped_chunks,
            self.pcm_queue.rejected_chunks,
            self.vad_worker.completed_segments,
            self.vad_worker.dropped_segments,
            self.whisper_worker.processed_segments,
            self.whisper_worker.empty_transcriptions,
            self.whisper_worker.failed_transcriptions,
            self.whisper_worker.callback_failures,
            self.transcription_publisher.published_results,
            self.transcription_publisher.rejected_results,
            self.transcription_publisher.failed_results,
        )
