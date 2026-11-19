"""Dedicated worker for assembling PCM chunks and detecting speech."""

from __future__ import annotations

import logging
import queue
import threading

from .audio_models import AudioSegment
from .audio_queue import PcmAudioQueue
from .frame_assembler import PcmFrameAssembler
from .streaming_vad import StreamingVad


LOGGER = logging.getLogger(__name__)


class VadWorker:
    """Process PCM chunks and publish completed AudioSegment objects."""

    def __init__(
        self,
        pcm_queue: PcmAudioQueue,
        frame_assembler: PcmFrameAssembler,
        streaming_vad: StreamingVad,
        *,
        segment_queue_size: int = 10,
    ) -> None:
        if segment_queue_size <= 0:
            raise ValueError(
                "segment_queue_size must be greater than zero"
            )

        self._pcm_queue = pcm_queue
        self._frame_assembler = frame_assembler
        self._streaming_vad = streaming_vad
        self._segment_queue: queue.Queue[AudioSegment] = queue.Queue(
            maxsize=segment_queue_size
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._shutdown_finalized = False
        self._drain = True
        self.completed_segments = 0
        self.dropped_segments = 0

    @property
    def is_running(self) -> bool:
        """Return whether the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the worker thread once."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._shutdown_finalized = False
            self._drain = True
            self._thread = threading.Thread(
                target=self._run,
                name="openduck-vad-worker",
                daemon=False,
            )
            thread = self._thread

        thread.start()

    def stop(
        self,
        *,
        drain: bool = True,
        timeout: float = 5.0,
    ) -> None:
        """Stop the worker, optionally draining queued PCM chunks."""
        with self._state_lock:
            already_stopping = self._stop_event.is_set()
            thread = self._thread
            if not already_stopping:
                self._drain = drain
                self._stop_event.set()

                unstarted_discard = thread is None and not drain
            else:
                unstarted_discard = False

        if unstarted_discard:
            self._pcm_queue.clear()
            self._frame_assembler.clear()
            self._streaming_vad.reset()
            with self._state_lock:
                self._shutdown_finalized = True

        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                LOGGER.warning(
                    "VadWorker thread did not stop within %.1f seconds",
                    timeout,
                )

    def get_segment(
        self,
        timeout: float = 0.1,
    ) -> AudioSegment | None:
        """Get a completed segment, or None when the timeout expires."""
        try:
            return self._segment_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def segment_task_done(self) -> None:
        """Mark a returned segment as processed."""
        self._segment_queue.task_done()

    @property
    def segment_queue_empty(self) -> bool:
        """Return whether no completed segments remain."""
        return self._segment_queue.empty()

    def clear_segments(self) -> int:
        """Remove queued segments and keep unfinished_tasks consistent."""
        removed = 0
        while True:
            try:
                self._segment_queue.get_nowait()
            except queue.Empty:
                break
            self._segment_queue.task_done()
            removed += 1
        return removed

    def _enqueue_segment(self, segment: AudioSegment) -> bool:
        """Try to add a segment without rejecting older segments."""
        try:
            self._segment_queue.put_nowait(segment)
        except queue.Full:
            self.dropped_segments += 1
            LOGGER.warning(
                "Segment queue full; dropped_segments=%d",
                self.dropped_segments,
            )
            return False

        self.completed_segments += 1
        return True

    def _run(self) -> None:
        """Consume PCM chunks until stopped and the queue is drained."""
        while True:
            with self._state_lock:
                stop_requested = self._stop_event.is_set()
                drain = self._drain

            if stop_requested and not drain:
                self._finalize_discard()
                break

            if (
                stop_requested
                and self._pcm_queue.empty
            ):
                break

            chunk = self._pcm_queue.get(timeout=0.1)
            if chunk is None:
                continue

            try:
                with self._state_lock:
                    process_chunk = not (
                        self._stop_event.is_set()
                        and not self._drain
                    )

                if process_chunk:
                    frames = self._frame_assembler.feed(chunk)
                    for frame in frames:
                        segment = (
                            self._streaming_vad.process_frame(frame)
                        )
                        if segment is not None:
                            self._enqueue_segment(segment)
            except Exception:
                LOGGER.exception("Failed to process PCM chunk")
            finally:
                self._pcm_queue.task_done()

        self._finalize_shutdown()

    def _finalize_discard(self) -> None:
        """Clear PCM and VAD state once for drain=False shutdown."""
        with self._state_lock:
            if self._shutdown_finalized:
                return
            self._shutdown_finalized = True

        self._pcm_queue.clear()
        self._frame_assembler.clear()
        self._streaming_vad.reset()

    def _finalize_shutdown(self) -> None:
        """Flush and clear VAD state once for drain=True shutdown."""
        with self._state_lock:
            if self._shutdown_finalized:
                return
            drain = self._drain
            self._shutdown_finalized = True

        if drain:
            self._frame_assembler.clear()
            segment = self._streaming_vad.flush()
            if segment is not None:
                self._enqueue_segment(segment)
