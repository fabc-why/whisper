"""Publish Whisper transcription results through rosbridge."""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    import roslibpy
except ModuleNotFoundError:
    roslibpy = None

from .audio_models import TranscriptionResult


LOGGER = logging.getLogger(__name__)

TRANSCRIPTION_TOPIC = "/openduck/speech/transcription"
TRANSCRIPTION_MESSAGE_TYPE = "std_msgs/String"


class TranscriptionPublisher:
    """Publish TranscriptionResult objects as JSON String messages."""

    def __init__(
        self,
        client: Any,
        topic: str = TRANSCRIPTION_TOPIC,
        message_type: str = TRANSCRIPTION_MESSAGE_TYPE,
    ) -> None:
        if roslibpy is None:
            raise ModuleNotFoundError(
                "roslibpy is required to publish transcriptions"
            )

        topic = topic.strip()
        if not topic:
            raise ValueError("transcription topic must not be empty")

        self.topic_name = topic
        self.message_type = message_type
        self.published_results = 0
        self.rejected_results = 0
        self.failed_results = 0
        self._cleaned_up = False

        self.topic = roslibpy.Topic(client, topic, message_type)
        self.topic.advertise()

    def publish(
        self,
        result: TranscriptionResult,
    ) -> bool:
        """Publish one result as a JSON std_msgs/String message."""
        text = result.text.strip()
        if not text:
            self.rejected_results += 1
            return False

        if self._cleaned_up:
            self.rejected_results += 1
            LOGGER.warning(
                "Cannot publish transcription after cleanup: segment_id=%s",
                result.segment_id,
            )
            return False

        payload = {
            "segment_id": result.segment_id,
            "text": text,
            "language": result.language,
            "started_at": result.started_at.isoformat(),
            "ended_at": result.ended_at.isoformat(),
        }
        json_string = json.dumps(payload, ensure_ascii=False)
        LOGGER.info("Publishing to ROS: %s", json_string)

        try:
            self.topic.publish(
                roslibpy.Message({"data": json_string})
            )
        except Exception:
            self.failed_results += 1
            LOGGER.exception(
                "Failed to publish transcription: segment_id=%s",
                result.segment_id,
            )
            return False

        self.published_results += 1
        return True

    def cleanup(self) -> None:
        """Stop advertising; do not terminate the shared ROS client."""
        if self._cleaned_up:
            return

        self._cleaned_up = True
        try:
            self.topic.unadvertise()
        except Exception:
            LOGGER.exception(
                "Failed to unadvertise transcription topic: %s",
                self.topic_name,
            )
