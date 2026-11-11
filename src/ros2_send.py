"""Publish recognized speech to ROS through rosbridge."""

from __future__ import annotations

import json
import os
from typing import Final

import roslibpy


TOPIC_NAME: Final = "openduck/whisper_heard"
MESSAGE_TYPE: Final = "std_msgs/String"


class Ros2Sender:
    """Publish text and its semantic command as one JSON string message."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        topic_name: str = TOPIC_NAME,
    ) -> None:
        self._ros = roslibpy.Ros(
            host=host or os.getenv("ROSBRIDGE_HOST", "localhost"),
            port=port or int(os.getenv("ROSBRIDGE_PORT", "9090")),
        )
        self._topic = roslibpy.Topic(
            self._ros,
            topic_name,
            MESSAGE_TYPE,
        )

    def connect(self) -> None:
        """Connect to rosbridge and advertise the speech topic."""
        self._ros.run()
        self._topic.advertise()

    def send(self, text: str, meaning: str) -> None:
        """Publish a JSON payload containing the raw text and its meaning."""
        payload = json.dumps(
            {"text": text, "meaning": meaning},
            ensure_ascii=False,
        )
        print(f"Publishing to ROS: {payload}")
        self._topic.publish(roslibpy.Message({"data": payload}))

    def close(self) -> None:
        """Stop advertising and close the rosbridge connection."""
        self._topic.unadvertise()
        self._ros.terminate()