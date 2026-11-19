"""Standalone entry point for the openduck_parts listening pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import roslibpy
from roslibpy.core import RosTimeoutError

if __package__:
    from .listening_service import ListeningService
else:
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from openduck_parts.listening_service import ListeningService


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    """Configuration for this standalone listening process."""

    host: str
    port: int
    model_name: str | None
    language: str | None
    device: str | None
    compute_type: str | None
    pcm_queue_size: int
    segment_queue_size: int
    drain_on_shutdown: bool


class WhisperApplication:
    """Own one rosbridge connection and one ListeningService."""

    def __init__(
        self,
        config: ApplicationConfig,
        *,
        ros_client: Any | None = None,
        service_factory: Callable[..., ListeningService] | None = None,
        ros_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._ros_client = ros_client
        self._ros_factory = ros_factory or (
            lambda host, port: roslibpy.Ros(host=host, port=port)
        )
        self._service_factory = service_factory or ListeningService
        self._service: ListeningService | None = None
        self._stop_event = threading.Event()
        self._cleaned_up = False
        self._state_lock = threading.Lock()

    def request_stop(self) -> None:
        """Set the shutdown request without performing heavy cleanup."""
        self._stop_event.set()

    def cleanup(self) -> None:
        """Stop the service and terminate the shared ROS client once."""
        with self._state_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
            service = self._service
            ros_client = self._ros_client

        if service is not None:
            try:
                service.stop(drain=self.config.drain_on_shutdown)
            except Exception:
                LOGGER.exception("ListeningService.stop() failed")

        if ros_client is not None:
            try:
                ros_client.terminate()
            except Exception:
                LOGGER.exception("Failed to terminate rosbridge client")

    def run(self) -> int:
        """Connect and run until signaled or the connection closes."""
        try:
            if self._ros_client is None:
                self._ros_client = self._ros_factory(
                    host=self.config.host,
                    port=self.config.port,
                )

            self._ros_client.run(timeout=10)
            self._service = self._service_factory(
                ros_client=self._ros_client,
                model_name=self.config.model_name,
                language=self.config.language,
                device=self.config.device,
                compute_type=self.config.compute_type,
                pcm_queue_size=self.config.pcm_queue_size,
                segment_queue_size=self.config.segment_queue_size,
            )
            self._service.start()

            while (
                not self._stop_event.is_set()
                and self._ros_client.is_connected
            ):
                time.sleep(0.1)

            if (
                not self._stop_event.is_set()
                and not self._ros_client.is_connected
            ):
                LOGGER.warning("rosbridge connection closed")
            return 0
        except RosTimeoutError as error:
            LOGGER.error("rosbridge connection timeout: %s", error)
            return 1
        except KeyboardInterrupt:
            LOGGER.info("Stopping Whisper application")
            return 0
        except Exception:
            LOGGER.exception("Whisper application failed")
            return 1
        finally:
            self.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standalone process arguments."""
    parser = argparse.ArgumentParser(
        description="Receive and transcribe external microphone audio.",
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
    parser.add_argument("--model", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--pcm-queue-size", type=int, default=150)
    parser.add_argument("--segment-queue-size", type=int, default=10)
    parser.add_argument("--no-drain", action="store_true")
    args = parser.parse_args(argv)

    if args.port <= 0:
        parser.error("--port must be greater than zero")
    if args.pcm_queue_size <= 0:
        parser.error("--pcm-queue-size must be greater than zero")
    if args.segment_queue_size <= 0:
        parser.error("--segment-queue-size must be greater than zero")
    return args


def build_config(argv: list[str] | None = None) -> ApplicationConfig:
    """Build ApplicationConfig from CLI and environment values."""
    args = parse_args(argv)
    return ApplicationConfig(
        host=args.host,
        port=args.port,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        pcm_queue_size=args.pcm_queue_size,
        segment_queue_size=args.segment_queue_size,
        drain_on_shutdown=not args.no_drain,
    )


def main(argv: list[str] | None = None) -> int:
    """Configure and run the standalone Whisper process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = build_config(argv)
    application = WhisperApplication(config)

    def handle_signal(signum: int, frame: Any) -> None:
        LOGGER.info("Received signal %d; stopping", signum)
        application.request_stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return application.run()


if __name__ == "__main__":
    raise SystemExit(main())
