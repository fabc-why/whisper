"""Dedicated worker for transcribing VadWorker audio segments."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .audio_models import TranscriptionResult
from .vad_worker import VadWorker
from .whisper_transcriber import WhisperTranscriber


LOGGER = logging.getLogger(__name__)

TranscriptionCallback = Callable[[TranscriptionResult], None]


class WhisperWorker:
    """Run Whisper transcription in one dedicated thread."""

    def __init__(
        self,
        vad_worker: VadWorker,
        transcriber: WhisperTranscriber,
        result_callback: TranscriptionCallback,
    ) -> None:
        self._vad_worker = vad_worker
        self._transcriber = transcriber
        self._result_callback = result_callback
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._drain = True

        self.processed_segments = 0
        self.empty_transcriptions = 0
        self.failed_transcriptions = 0
        self.callback_failures = 0

    @property
    def is_running(self) -> bool:
        """Return whether the Whisper worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker thread once."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._drain = True
            self._thread = threading.Thread(
                target=self._run,
                name="openduck-whisper-worker",
                daemon=False,
            )
            thread = self._thread

        thread.start()

    def stop(
        self,
        *,
        drain: bool = True,
        timeout: float = 15.0,
    ) -> None:
        """Stop the worker, optionally draining completed segments."""
        with self._state_lock:
            already_stopping = self._stop_event.is_set()
            thread = self._thread
            if not already_stopping:
                self._stop_event.set()
                self._drain = drain

            if not already_stopping and not drain:
                self._vad_worker.clear_segments()

        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                LOGGER.warning(
                    "WhisperWorker thread did not stop within "
                    "%.1f seconds",
                    timeout,
                )

    def _run(self) -> None:
        """Consume and transcribe segments until shutdown is safe."""
        while True:
            with self._state_lock:
                stop_requested = self._stop_event.is_set()
                drain = self._drain

            if stop_requested and not drain:
                break

            if (
                stop_requested
                and self._vad_worker.segment_queue_empty
            ):
                break

            segment = self._vad_worker.get_segment(timeout=0.1)
            if segment is None:
                continue

            with self._state_lock:
                stop_requested = self._stop_event.is_set()
                drain = self._drain
            if stop_requested and not drain:
                self._vad_worker.segment_task_done()
                continue

            try:
                result = self._transcriber.transcribe(segment)
                if not result.text.strip():
                    self.empty_transcriptions += 1
                    continue

                try:
                    self._result_callback(result)
                except Exception:
                    self.callback_failures += 1
                    LOGGER.exception(
                        "Transcription callback failed for segment_id=%s",
                        segment.segment_id,
                    )
                else:
                    self.processed_segments += 1
            except Exception:
                self.failed_transcriptions += 1
                LOGGER.exception(
                    "Whisper transcription failed for segment_id=%s",
                    segment.segment_id,
                )
            finally:
                self._vad_worker.segment_task_done()
