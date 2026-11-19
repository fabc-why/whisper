#!/usr/bin/env python3

# openduckのロボット側で動かすコードです。
# シミュレーション内にマイクを置けないので、PC側のマイクを使って音声入力をしています。

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import queue
import signal
import sys
import time
from dataclasses import dataclass

import roslibpy
import sounddevice as sd
from roslibpy.core import RosTimeoutError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Microphone capture and rosbridge configuration."""

    host: str = "localhost"
    port: int = 9090
    sample_rate: int = 16_000
    channels: int = 1
    block_duration_ms: int = 20
    device: int | str | None = None
    audio_topic: str = "/openduck/audio/raw"
    info_topic: str = "/openduck/audio/info"
    queue_size: int = 100

    @property
    def frames_per_block(self) -> int:
        return int(self.sample_rate * self.block_duration_ms / 1000)


class MicrophonePublisher:
    """Capture microphone PCM audio and publish base64 strings via rosbridge."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.audio_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=config.queue_size
        )
        self.stream: sd.RawInputStream | None = None
        self.running = False
        self.stopped = False
        self.dropped_blocks = 0
        self.published_blocks = 0

        self.ros = roslibpy.Ros(
            host=config.host,
            port=config.port,
        )
        self.audio_topic = roslibpy.Topic(
            self.ros,
            config.audio_topic,
            "std_msgs/String",
        )
        self.info_topic = roslibpy.Topic(
            self.ros,
            config.info_topic,
            "std_msgs/String",
        )

    def connect(self) -> None:
        """Connect to rosbridge and advertise String topics."""
        LOGGER.info(
            "Connecting to rosbridge ws://%s:%d",
            self.config.host,
            self.config.port,
        )
        self.ros.run(timeout=10)
        self.audio_topic.advertise()
        self.info_topic.advertise()
        LOGGER.info("Connected to rosbridge")

    def start(self) -> None:
        """Start microphone capture and begin publishing queued blocks."""
        if self.running:
            return

        self._publish_audio_info()
        self.stream = sd.RawInputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.frames_per_block,
            device=self.config.device,
            channels=self.config.channels,
            dtype="int16",
            callback=self._audio_callback,
        )
        self.stream.start()
        self.running = True
        self.ros.call_later(0.005, self._publish_pending_audio)

        LOGGER.info(
            "Microphone started: device=%r, rate=%d Hz, channels=%d, "
            "block=%d ms, frames=%d, audio_topic=%s",
            self.config.device,
            self.config.sample_rate,
            self.config.channels,
            self.config.block_duration_ms,
            self.config.frames_per_block,
            self.config.audio_topic,
        )

    def _audio_callback(
        self,
        indata,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """Enqueue microphone PCM without doing ROS/network work."""
        if status:
            LOGGER.warning("Microphone callback status: %s", status)
        if not self.running:
            return

        if frames != self.config.frames_per_block:
            LOGGER.debug(
                "Unexpected audio frame count: expected=%d actual=%d",
                self.config.frames_per_block,
                frames,
            )

        audio_bytes = bytes(indata)
        try:
            self.audio_queue.put_nowait(audio_bytes)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                pass

            try:
                self.audio_queue.put_nowait(audio_bytes)
            except queue.Full:
                pass

            self.dropped_blocks += 1
            if self.dropped_blocks % 100 == 1:
                LOGGER.warning(
                    "Audio queue overflow: dropped_blocks=%d",
                    self.dropped_blocks,
                )

    def _publish_pending_audio(self) -> None:
        """Publish queued audio blocks as base64 std_msgs/String messages."""
        if not self.running:
            return

        for _ in range(10):
            try:
                audio_bytes = self.audio_queue.get_nowait()
            except queue.Empty:
                break

            try:
                encoded = base64.b64encode(audio_bytes).decode("ascii")
                self.audio_topic.publish(
                    roslibpy.Message({"data": encoded})
                )
                self.published_blocks += 1
            finally:
                self.audio_queue.task_done()

        self.ros.call_later(0.005, self._publish_pending_audio)

    def _publish_audio_info(self) -> None:
        """Publish stream metadata as a JSON std_msgs/String message."""
        info = {
            "channels": self.config.channels,
            "sample_rate": self.config.sample_rate,
            "sample_format": "S16LE",
            "bitrate": (
                self.config.sample_rate
                * 16
                * self.config.channels
            ),
            "coding_format": "PCM",
            "encoding": "base64",
        }
        self.info_topic.publish(
            roslibpy.Message(
                {"data": json.dumps(info, ensure_ascii=False)}
            )
        )

    def stop(self) -> None:
        """Stop capture and close rosbridge."""
        if self.stopped:
            return
        self.stopped = True
        self.running = False

        stream = self.stream
        self.stream = None
        if stream is not None:
            try:
                if stream.active:
                    stream.stop()
            except sd.PortAudioError as error:
                LOGGER.warning("Failed to stop microphone stream: %s", error)
            try:
                stream.close()
            except sd.PortAudioError as error:
                LOGGER.warning("Failed to close microphone stream: %s", error)

        while True:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break

        try:
            self.audio_topic.unadvertise()
        finally:
            try:
                self.info_topic.unadvertise()
            finally:
                self.ros.terminate()

        LOGGER.info(
            "Microphone stopped: published_blocks=%d dropped_blocks=%d",
            self.published_blocks,
            self.dropped_blocks,
        )


def list_audio_devices() -> None:
    """Print available PortAudio devices."""
    print(sd.query_devices())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture microphone audio and publish base64 PCM via rosbridge."
        )
    )
    parser.add_argument(
        "--host",
        default=os.getenv("ROSBRIDGE_HOST", "localhost"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ROSBRIDGE_PORT", "9090")),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Input device index or name.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--block-ms", type=int, default=20)
    parser.add_argument(
        "--audio-topic",
        default="/openduck/audio/raw",
    )
    parser.add_argument(
        "--info-topic",
        default="/openduck/audio/info",
    )
    parser.add_argument("--queue-size", type=int, default=100)
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


def normalize_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    if args.list_devices:
        list_audio_devices()
        return 0
    if args.port <= 0:
        raise ValueError("--port must be greater than zero")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be greater than zero")
    if args.channels <= 0:
        raise ValueError("--channels must be greater than zero")
    if args.block_ms <= 0:
        raise ValueError("--block-ms must be greater than zero")
    if args.queue_size <= 0:
        raise ValueError("--queue-size must be greater than zero")

    config = AudioConfig(
        host=args.host,
        port=args.port,
        sample_rate=args.sample_rate,
        channels=args.channels,
        block_duration_ms=args.block_ms,
        device=normalize_device(args.device),
        audio_topic=args.audio_topic,
        info_topic=args.info_topic,
        queue_size=args.queue_size,
    )
    publisher = MicrophonePublisher(config)
    stopping = False

    def request_shutdown(signum, frame) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        LOGGER.info("Received signal %d; shutting down", signum)
        publisher.stop()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    try:
        publisher.connect()
        publisher.start()
        while publisher.ros.is_connected:
            time.sleep(0.1)
        LOGGER.warning("rosbridge connection closed")
    except KeyboardInterrupt:
        pass
    except RosTimeoutError as error:
        LOGGER.error("rosbridge connection timeout: %s", error)
        return 1
    except sd.PortAudioError as error:
        LOGGER.error("PortAudio error: %s", error)
        return 1
    finally:
        publisher.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
