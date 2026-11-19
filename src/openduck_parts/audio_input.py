"""ROS2 audio input subscriber."""

from __future__ import annotations

import base64
import binascii
import logging
from collections.abc import Callable
from typing import Any

try:
    import roslibpy
except ModuleNotFoundError:
    roslibpy = None


RAW_AUDIO_TOPIC = '/openduck/audio/raw'
RAW_AUDIO_TYPE = 'std_msgs/String'

AudioCallback = Callable[[bytes], None]


def decode_audio_message(message: Any) -> bytes:
    """Convert a ROS bridge audio message into raw bytes."""
    if isinstance(message, dict):
        value = message.get('data')
    else:
        value = getattr(message, 'data', None)

    if not isinstance(value, str):
        raise ValueError('audio message data must be a base64 string')

    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError('audio data is not valid base64') from error


class RawAudioSubscriber:
    """Subscribe to external microphone audio through ROS."""

    def __init__(
        self,
        client: Any,
        callback: AudioCallback | None = None,
        topic: str = RAW_AUDIO_TOPIC,
        message_type: str = RAW_AUDIO_TYPE,
    ) -> None:
        if roslibpy is None:
            raise ModuleNotFoundError(
                'roslibpy is required to subscribe to ROS audio'
            )
        self.topic_name = topic.strip()
        if not self.topic_name:
            raise ValueError('audio topic must not be empty')

        self.callback = callback
        self.topic = roslibpy.Topic(client, self.topic_name, message_type)
        self.topic.subscribe(self._handle_message)
        self.cleaned_up = False

    def _handle_message(self, message: Any) -> None:
        try:
            audio = decode_audio_message(message)
        except ValueError:
            logging.exception('failed to decode raw audio message')
            return

        if self.callback is not None:
            try:
                self.callback(audio)
            except Exception:
                logging.exception('raw audio callback failed')

    def cleanup(self) -> None:
        """Unsubscribe from the raw audio topic."""
        if self.cleaned_up:
            return
        self.cleaned_up = True
        self.topic.unsubscribe()
